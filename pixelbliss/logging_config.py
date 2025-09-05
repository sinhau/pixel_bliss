import logging
import sys
from typing import Optional
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for terminal output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Format the message
        formatted = super().format(record)
        return formatted

class ProgressLogger:
    """Special logger for tracking pipeline progress with visual indicators."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._step_count = 0
        self._total_steps = 0
        self._current_operation = None
        self._operation_progress = {}
    
    def start_pipeline(self, total_steps: int):
        """Initialize pipeline progress tracking."""
        self._step_count = 0
        self._total_steps = total_steps
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.BLUE}🚀 PIXELBLISS PIPELINE STARTED{Style.RESET_ALL}")
        self.logger.info(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    
    def step(self, description: str, details: Optional[str] = None):
        """Log a pipeline step with progress indicator."""
        self._step_count += 1
        progress = f"[{self._step_count}/{self._total_steps}]"
        
        # Create visual progress bar
        bar_length = 20
        filled = int((self._step_count / self._total_steps) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        message = f"{Fore.CYAN}{progress}{Style.RESET_ALL} {Fore.GREEN}{bar}{Style.RESET_ALL} {description}"
        if details:
            message += f" {Fore.WHITE}({details}){Style.RESET_ALL}"
        
        self.logger.info(message)
    
    def substep(self, description: str, details: Optional[str] = None):
        """Log a sub-step within a main pipeline step."""
        message = f"  {Fore.YELLOW}▶{Style.RESET_ALL} {description}"
        if details:
            message += f" {Fore.WHITE}({details}){Style.RESET_ALL}"
        self.logger.info(message)
    
    def success(self, description: str, details: Optional[str] = None):
        """Log a successful operation."""
        message = f"  {Fore.GREEN}✓{Style.RESET_ALL} {description}"
        if details:
            message += f" {Fore.WHITE}({details}){Style.RESET_ALL}"
        self.logger.info(message)
    
    def warning(self, description: str, details: Optional[str] = None):
        """Log a warning."""
        message = f"  {Fore.YELLOW}⚠{Style.RESET_ALL} {description}"
        if details:
            message += f" {Fore.WHITE}({details}){Style.RESET_ALL}"
        self.logger.warning(message)
    
    def error(self, description: str, details: Optional[str] = None):
        """Log an error."""
        message = f"  {Fore.RED}✗{Style.RESET_ALL} {description}"
        if details:
            message += f" {Fore.WHITE}({details}){Style.RESET_ALL}"
        self.logger.error(message)
    
    def start_operation(self, operation_name: str, total_items: int, description: str = None):
        """Start tracking progress for a parallel operation."""
        self._current_operation = operation_name
        self._operation_progress[operation_name] = {
            'completed': 0,
            'total': total_items,
            'description': description or operation_name
        }
        
        desc = description or operation_name
        message = f"  {Fore.BLUE}🔄{Style.RESET_ALL} Starting {desc} ({total_items} items)"
        self.logger.info(message)
    
    def log_base_prompt_generation(self, category: str, provider: str, model: str):
        """Log the start of base prompt generation."""
        message = f"  {Fore.CYAN}🎯{Style.RESET_ALL} Generating base prompt for category '{category}' using {provider}/{model}"
        self.logger.info(message)
    
    def log_base_knobs_selected(self, base_knobs: dict):
        """Log the selected base knobs for prompt generation."""
        message = f"  {Fore.BLUE}🎛️{Style.RESET_ALL} Base knobs selected:"
        self.logger.info(message)
        for knob_name, knob_value in base_knobs.items():
            knob_msg = f"    {Fore.YELLOW}•{Style.RESET_ALL} {Fore.CYAN}{knob_name}{Style.RESET_ALL}: {Fore.WHITE}{knob_value}{Style.RESET_ALL}"
            self.logger.info(knob_msg)
    
    def log_base_prompt_success(self, base_prompt: str, generation_time: float = None):
        """Log successful base prompt generation."""
        preview = base_prompt[:80] + "..." if len(base_prompt) > 80 else base_prompt
        time_info = f" ({generation_time:.2f}s)" if generation_time else ""
        message = f"  {Fore.GREEN}✓{Style.RESET_ALL} Base prompt generated{time_info}: {Fore.WHITE}{preview}{Style.RESET_ALL}"
        self.logger.info(message)
    
    def log_variant_prompt_generation_start(self, num_variants: int, provider: str, model: str, async_mode: bool):
        """Log the start of variant prompt generation."""
        mode_text = "parallel" if async_mode else "sequential"
        message = f"  {Fore.CYAN}🔀{Style.RESET_ALL} Generating {num_variants} prompt variants using {provider}/{model} ({mode_text} mode)"
        self.logger.info(message)
    
    def log_variant_knobs_selected(self, variant_knobs_list: list, strategy: str):
        """Log the selected variant knobs for prompt generation."""
        strategy_text = "single knob variation" if strategy == "single" else "full knob variation"
        message = f"  {Fore.BLUE}🎛️{Style.RESET_ALL} Variant knobs selected ({strategy_text}):"
        self.logger.info(message)
        
        for i, variant_knobs in enumerate(variant_knobs_list, 1):
            variant_header = f"    {Fore.MAGENTA}Variant #{i}:{Style.RESET_ALL}"
            self.logger.info(variant_header)
            for knob_name, knob_value in variant_knobs.items():
                knob_msg = f"      {Fore.YELLOW}•{Style.RESET_ALL} {Fore.CYAN}{knob_name}{Style.RESET_ALL}: {Fore.WHITE}{knob_value}{Style.RESET_ALL}"
                self.logger.info(knob_msg)
    
    def log_variant_prompt_success(self, variant_prompts: list, generation_time: float = None):
        """Log successful variant prompt generation."""
        time_info = f" ({generation_time:.2f}s)" if generation_time else ""
        message = f"  {Fore.GREEN}✓{Style.RESET_ALL} Generated {len(variant_prompts)} prompt variants{time_info}"
        self.logger.info(message)
        
        # Log each variant with a preview
        for i, variant in enumerate(variant_prompts, 1):
            preview = variant[:60] + "..." if len(variant) > 60 else variant
            variant_msg = f"    {Fore.YELLOW}#{i}{Style.RESET_ALL} {Fore.WHITE}{preview}{Style.RESET_ALL}"
            self.logger.debug(variant_msg)
    
    def log_variant_prompt_error(self, error: str, generation_time: float = None):
        """Log variant prompt generation error."""
        time_info = f" (after {generation_time:.2f}s)" if generation_time else ""
        message = f"  {Fore.RED}✗{Style.RESET_ALL} Variant prompt generation failed{time_info}: {error}"
        self.logger.error(message)
    
    def update_operation_progress(self, operation_name: str, completed: int = None, increment: int = 1):
        """Update progress for a parallel operation."""
        if operation_name not in self._operation_progress:
            return
        
        progress = self._operation_progress[operation_name]
        
        if completed is not None:
            progress['completed'] = completed
        else:
            progress['completed'] += increment
        
        # Create progress bar for this operation
        total = progress['total']
        current = min(progress['completed'], total)  # Ensure we don't exceed total
        percentage = (current / total) * 100 if total > 0 else 0
        
        # Visual progress bar
        bar_length = 15
        filled = int((current / total) * bar_length) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Status indicator
        if current == total:
            status_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        else:
            status_icon = f"{Fore.YELLOW}⏳{Style.RESET_ALL}"
        
        message = f"  {status_icon} {progress['description']}: {Fore.CYAN}{bar}{Style.RESET_ALL} {current}/{total} ({percentage:.0f}%)"
        self.logger.info(message)
    
    def finish_operation(self, operation_name: str, success: bool = True):
        """Finish tracking progress for a parallel operation."""
        if operation_name not in self._operation_progress:
            return
        
        progress = self._operation_progress[operation_name]
        
        if success:
            # Ensure we show 100% completion
            self.update_operation_progress(operation_name, completed=progress['total'])
            message = f"  {Fore.GREEN}✅{Style.RESET_ALL} {progress['description']} completed successfully"
        else:
            message = f"  {Fore.RED}❌{Style.RESET_ALL} {progress['description']} failed"
        
        self.logger.info(message)
        
        # Clean up if this was the current operation
        if self._current_operation == operation_name:
            self._current_operation = None
    
    def finish_pipeline(self, success: bool = True):
        """Finish pipeline progress tracking."""
        if success:
            self.logger.info(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            self.logger.info(f"{Fore.GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY{Style.RESET_ALL}")
            self.logger.info(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        else:
            self.logger.error(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
            self.logger.error(f"{Fore.RED}💥 PIPELINE FAILED{Style.RESET_ALL}")
            self.logger.error(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_colors: bool = True
) -> tuple[logging.Logger, ProgressLogger]:
    """
    Set up centralized logging configuration for PixelBliss.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        enable_colors: Whether to enable colored output for console
        
    Returns:
        tuple: (main_logger, progress_logger)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create main logger
    logger = logging.getLogger('pixelbliss')
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_colors:
        console_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        console_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
    else:
        console_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        console_formatter = logging.Formatter(console_format, datefmt='%H:%M:%S')
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        file_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Create progress logger
    progress_logger = ProgressLogger(logger)
    
    return logger, progress_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    
    Args:
        name: Name of the module/component
        
    Returns:
        logging.Logger: Child logger instance
    """
    return logging.getLogger(f'pixelbliss.{name}')
