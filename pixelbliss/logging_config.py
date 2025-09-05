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
