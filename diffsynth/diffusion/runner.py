import contextlib, os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


class _NullProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def step(self):
        return None


def _record_function(enabled: bool, name: str):
    if enabled:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _resolve_profiler_activities(activity_names: str):
    available_activities = {"cpu": torch.profiler.ProfilerActivity.CPU}
    if hasattr(torch.profiler.ProfilerActivity, "CUDA") and torch.cuda.is_available():
        available_activities["cuda"] = torch.profiler.ProfilerActivity.CUDA

    requested = [name.strip().lower() for name in activity_names.split(",") if name.strip()]
    if not requested:
        raise ValueError("At least one profiler activity must be provided.")

    activities = []
    unsupported = []
    for name in requested:
        activity = available_activities.get(name)
        if activity is None:
            unsupported.append(name)
        elif activity not in activities:
            activities.append(activity)

    if unsupported:
        supported = ",".join(sorted(available_activities))
        raise ValueError(f"Unsupported profiler activities: {', '.join(unsupported)}. Supported values: {supported}.")

    return activities


def _build_profiler(accelerator: Accelerator, model_logger: ModelLogger, args):
    if args is None or not getattr(args, "enable_profiler", False):
        return _NullProfiler(), False

    if not getattr(args, "profiler_all_processes", False) and not accelerator.is_main_process:
        return _NullProfiler(), False

    wait_steps = max(0, getattr(args, "profiler_wait_steps", 1))
    warmup_steps = max(0, getattr(args, "profiler_warmup_steps", 1))
    active_steps = getattr(args, "profiler_active_steps", 3)
    repeat = getattr(args, "profiler_repeat", 1)
    if active_steps <= 0:
        raise ValueError("--profiler_active_steps must be greater than 0.")
    if repeat <= 0:
        raise ValueError("--profiler_repeat must be greater than 0.")

    trace_root = getattr(args, "profiler_trace_dir", None) or os.path.join(model_logger.output_path, "profiler")
    trace_dir = os.path.join(trace_root, f"rank{accelerator.process_index:02d}")
    activities = _resolve_profiler_activities(getattr(args, "profiler_activities", "cpu,cuda"))
    trace_index = 0

    def on_trace_ready(profiler):
        nonlocal trace_index
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"perfetto_trace_{trace_index:02d}_step_{profiler.step_num:05d}.json")
        profiler.export_chrome_trace(trace_path)
        print(f"[profiler][rank {accelerator.process_index}] exported Perfetto trace to {trace_path}", flush=True)
        trace_index += 1

    if accelerator.is_main_process:
        target = "all ranks" if getattr(args, "profiler_all_processes", False) else "rank 0"
        accelerator.print(
            f"[profiler] capturing {target} to {trace_root} "
            f"(wait={wait_steps}, warmup={warmup_steps}, active={active_steps}, repeat={repeat})"
        )

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=repeat,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=getattr(args, "profiler_record_shapes", False),
        profile_memory=getattr(args, "profiler_profile_memory", False),
        with_stack=getattr(args, "profiler_with_stack", False),
        with_flops=getattr(args, "profiler_with_flops", False),
    )
    return profiler, True


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    profiler, profiler_enabled = _build_profiler(accelerator, model_logger, args)
    with profiler:
        for epoch_id in range(num_epochs):
            data_iterator = iter(dataloader)
            for _ in tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process):
                with _record_function(profiler_enabled, "dataloader_next"):
                    data = next(data_iterator)
                with _record_function(profiler_enabled, "train_step"):
                    with accelerator.accumulate(model):
                        with _record_function(profiler_enabled, "optimizer_zero_grad"):
                            optimizer.zero_grad()
                        with _record_function(profiler_enabled, "forward"):
                            if dataset.load_from_cache:
                                loss = model({}, inputs=data)
                            else:
                                loss = model(data)
                        with _record_function(profiler_enabled, "backward"):
                            accelerator.backward(loss)
                        with _record_function(profiler_enabled, "optimizer_step"):
                            optimizer.step()
                        with _record_function(profiler_enabled, "logger_step"):
                            model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                        with _record_function(profiler_enabled, "scheduler_step"):
                            scheduler.step()
                profiler.step()
            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
