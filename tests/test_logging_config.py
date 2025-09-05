import pytest
from unittest.mock import Mock, patch
from pixelbliss.logging_config import ProgressLogger, ColoredFormatter, setup_logging, get_logger


class TestColoredFormatter:
    """Test ColoredFormatter class."""

    def test_colored_formatter_with_colors(self):
        """Test ColoredFormatter adds colors to log levels."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Create a proper log record
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Test the actual formatter behavior
        result = formatter.format(record)
        
        # Should contain ANSI color codes (either \033[ or \x1b[)
        assert ("\033[" in result or "\x1b[" in result)  # ANSI escape sequence
        assert "INFO" in result

    def test_colored_formatter_unknown_level(self):
        """Test ColoredFormatter with unknown log level."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Create a mock log record with unknown level
        record = Mock()
        record.levelname = "UNKNOWN"
        record.getMessage.return_value = "Test message"
        
        # Mock the parent format method
        with patch.object(ColoredFormatter.__bases__[0], 'format', return_value="UNKNOWN - Test message") as mock_format:
            result = formatter.format(record)
            
            # Should not modify unknown level
            assert result == "UNKNOWN - Test message"
            mock_format.assert_called_once_with(record)


class TestProgressLogger:
    """Test ProgressLogger class."""

    def test_progress_logger_init(self):
        """Test ProgressLogger initialization."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        assert progress_logger.logger == mock_logger
        assert progress_logger._step_count == 0
        assert progress_logger._total_steps == 0
        assert progress_logger._current_operation is None
        assert progress_logger._operation_progress == {}

    def test_start_pipeline(self):
        """Test start_pipeline method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.start_pipeline(5)
        
        assert progress_logger._total_steps == 5
        assert progress_logger._step_count == 0
        assert mock_logger.info.call_count == 3  # Header, title, footer

    def test_step_progress(self):
        """Test step method with progress tracking."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_pipeline(3)
        
        progress_logger.step("Test step", "details")
        
        assert progress_logger._step_count == 1
        mock_logger.info.assert_called()
        
        # Check that the call contains progress indicators
        call_args = mock_logger.info.call_args_list[-1][0][0]
        assert "[1/3]" in call_args
        assert "Test step" in call_args
        assert "details" in call_args

    def test_substep(self):
        """Test substep method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.substep("Sub operation", "sub details")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "▶" in call_args
        assert "Sub operation" in call_args
        assert "sub details" in call_args

    def test_success(self):
        """Test success method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.success("Operation completed", "success details")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "Operation completed" in call_args
        assert "success details" in call_args

    def test_warning(self):
        """Test warning method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.warning("Warning message", "warning details")
        
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0][0]
        assert "⚠" in call_args
        assert "Warning message" in call_args
        assert "warning details" in call_args

    def test_error(self):
        """Test error method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.error("Error message", "error details")
        
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "✗" in call_args
        assert "Error message" in call_args
        assert "error details" in call_args

    def test_start_operation(self):
        """Test start_operation method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.start_operation("test_op", 5, "Test Operation")
        
        assert progress_logger._current_operation == "test_op"
        assert "test_op" in progress_logger._operation_progress
        assert progress_logger._operation_progress["test_op"]["total"] == 5
        assert progress_logger._operation_progress["test_op"]["completed"] == 0
        assert progress_logger._operation_progress["test_op"]["description"] == "Test Operation"
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "🔄" in call_args
        assert "Test Operation" in call_args
        assert "5 items" in call_args

    def test_start_operation_without_description(self):
        """Test start_operation method without description."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.start_operation("test_op", 3)
        
        assert progress_logger._operation_progress["test_op"]["description"] == "test_op"

    def test_update_operation_progress_increment(self):
        """Test update_operation_progress with increment."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 5, "Test Operation")
        
        progress_logger.update_operation_progress("test_op")
        
        assert progress_logger._operation_progress["test_op"]["completed"] == 1
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "1/5" in call_args
        assert "20%" in call_args

    def test_update_operation_progress_set_completed(self):
        """Test update_operation_progress with explicit completed value."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 5, "Test Operation")
        
        progress_logger.update_operation_progress("test_op", completed=3)
        
        assert progress_logger._operation_progress["test_op"]["completed"] == 3
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "3/5" in call_args
        assert "60%" in call_args

    def test_update_operation_progress_custom_increment(self):
        """Test update_operation_progress with custom increment."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 10, "Test Operation")
        
        progress_logger.update_operation_progress("test_op", increment=3)
        
        assert progress_logger._operation_progress["test_op"]["completed"] == 3

    def test_update_operation_progress_nonexistent_operation(self):
        """Test update_operation_progress with nonexistent operation."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        # Should not raise error
        progress_logger.update_operation_progress("nonexistent")
        
        # Logger should not be called
        mock_logger.info.assert_not_called()

    def test_update_operation_progress_completion(self):
        """Test update_operation_progress when operation completes."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 3, "Test Operation")
        
        progress_logger.update_operation_progress("test_op", completed=3)
        
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args  # Should show completion checkmark
        assert "100%" in call_args

    def test_update_operation_progress_exceeds_total(self):
        """Test update_operation_progress when completed exceeds total."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 3, "Test Operation")
        
        progress_logger.update_operation_progress("test_op", completed=5)
        
        # Should clamp to total
        call_args = mock_logger.info.call_args[0][0]
        assert "3/3" in call_args
        assert "100%" in call_args

    def test_update_operation_progress_zero_total(self):
        """Test update_operation_progress with zero total."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 0, "Test Operation")
        
        progress_logger.update_operation_progress("test_op")
        
        call_args = mock_logger.info.call_args[0][0]
        assert "0/0" in call_args  # Should clamp to total (0)
        assert "0%" in call_args  # Should handle division by zero

    def test_finish_operation_success(self):
        """Test finish_operation with success."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 3, "Test Operation")
        
        progress_logger.finish_operation("test_op", success=True)
        
        # Should update progress to 100% and show success
        assert mock_logger.info.call_count >= 2  # start + updates + finish
        call_args = mock_logger.info.call_args[0][0]
        assert "✅" in call_args
        assert "completed successfully" in call_args

    def test_finish_operation_failure(self):
        """Test finish_operation with failure."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 3, "Test Operation")
        
        progress_logger.finish_operation("test_op", success=False)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "❌" in call_args
        assert "failed" in call_args

    def test_finish_operation_nonexistent(self):
        """Test finish_operation with nonexistent operation."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        # Should not raise error
        progress_logger.finish_operation("nonexistent")
        
        # Logger should not be called
        mock_logger.info.assert_not_called()

    def test_finish_operation_clears_current(self):
        """Test finish_operation clears current operation."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_operation("test_op", 3, "Test Operation")
        
        assert progress_logger._current_operation == "test_op"
        
        progress_logger.finish_operation("test_op")
        
        assert progress_logger._current_operation is None

    def test_finish_pipeline_success(self):
        """Test finish_pipeline with success."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.finish_pipeline(success=True)
        
        assert mock_logger.info.call_count == 3  # Header, message, footer
        # Check for success indicators
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        success_message = calls[1]
        assert "🎉" in success_message
        assert "COMPLETED SUCCESSFULLY" in success_message

    def test_finish_pipeline_failure(self):
        """Test finish_pipeline with failure."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.finish_pipeline(success=False)
        
        assert mock_logger.error.call_count == 3  # Header, message, footer
        # Check for failure indicators
        calls = [call[0][0] for call in mock_logger.error.call_args_list]
        failure_message = calls[1]
        assert "💥" in failure_message
        assert "PIPELINE FAILED" in failure_message


class TestSetupLogging:
    """Test setup_logging function."""

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_setup_logging_basic(self, mock_get_logger):
        """Test basic setup_logging functionality."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger, progress_logger = setup_logging()
        
        assert logger == mock_logger
        assert isinstance(progress_logger, ProgressLogger)
        mock_logger.setLevel.assert_called()
        mock_logger.addHandler.assert_called()

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_setup_logging_with_file(self, mock_get_logger):
        """Test setup_logging with file output."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with patch('pixelbliss.logging_config.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            
            with patch('pixelbliss.logging_config.logging.FileHandler') as mock_file_handler:
                logger, progress_logger = setup_logging(log_file="test.log")
                
                mock_file_handler.assert_called_once_with("test.log")
                mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_setup_logging_no_colors(self, mock_get_logger):
        """Test setup_logging without colors."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger, progress_logger = setup_logging(enable_colors=False)
        
        assert logger == mock_logger
        assert isinstance(progress_logger, ProgressLogger)

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_setup_logging_debug_level(self, mock_get_logger):
        """Test setup_logging with DEBUG level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger, progress_logger = setup_logging(level="DEBUG")
        
        # Should set DEBUG level (10)
        mock_logger.setLevel.assert_called_with(10)

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_setup_logging_invalid_level(self, mock_get_logger):
        """Test setup_logging with invalid level defaults to INFO."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger, progress_logger = setup_logging(level="INVALID")
        
        # Should default to INFO level (20)
        mock_logger.setLevel.assert_called_with(20)


class TestGetLogger:
    """Test get_logger function."""

    @patch('pixelbliss.logging_config.logging.getLogger')
    def test_get_logger(self, mock_get_logger):
        """Test get_logger returns child logger."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        result = get_logger("test_module")
        
        mock_get_logger.assert_called_once_with("pixelbliss.test_module")
        assert result == mock_logger


class TestProgressLoggerIntegration:
    """Test ProgressLogger integration scenarios."""

    def test_multiple_operations(self):
        """Test handling multiple operations."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        # Start first operation
        progress_logger.start_operation("op1", 3, "Operation 1")
        progress_logger.update_operation_progress("op1")
        
        # Start second operation
        progress_logger.start_operation("op2", 2, "Operation 2")
        progress_logger.update_operation_progress("op2")
        
        # Both operations should be tracked
        assert "op1" in progress_logger._operation_progress
        assert "op2" in progress_logger._operation_progress
        assert progress_logger._current_operation == "op2"
        
        # Finish operations
        progress_logger.finish_operation("op1")
        progress_logger.finish_operation("op2")
        
        assert progress_logger._current_operation is None

    def test_operation_progress_edge_cases(self):
        """Test operation progress edge cases."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        # Test with single item
        progress_logger.start_operation("single", 1, "Single Item")
        progress_logger.update_operation_progress("single")
        
        call_args = mock_logger.info.call_args[0][0]
        assert "1/1" in call_args
        assert "100%" in call_args
        assert "✓" in call_args  # Should show completion

    def test_substep_without_details(self):
        """Test substep without details."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.substep("Sub operation")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "▶" in call_args
        assert "Sub operation" in call_args
        # Should not contain parentheses when no details
        assert "(" not in call_args

    def test_success_without_details(self):
        """Test success without details."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.success("Operation completed")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "Operation completed" in call_args
        # Should not contain parentheses when no details
        assert "(" not in call_args

    def test_warning_without_details(self):
        """Test warning without details."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.warning("Warning message")
        
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0][0]
        assert "⚠" in call_args
        assert "Warning message" in call_args
        # Should not contain parentheses when no details
        assert "(" not in call_args

    def test_error_without_details(self):
        """Test error without details."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.error("Error message")
        
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "✗" in call_args
        assert "Error message" in call_args
        # Should not contain parentheses when no details
        assert "(" not in call_args

    def test_step_without_details(self):
        """Test step without details."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        progress_logger.start_pipeline(3)
        
        progress_logger.step("Test step")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "[1/3]" in call_args
        assert "Test step" in call_args
        # Should not contain parentheses when no details
        assert "(" not in call_args

    def test_log_base_prompt_generation(self):
        """Test log_base_prompt_generation method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.log_base_prompt_generation("sci-fi", "openai", "gpt-5")
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "🎯" in call_args
        assert "sci-fi" in call_args
        assert "openai/gpt-5" in call_args
        assert "Generating base prompt" in call_args

    def test_log_base_prompt_success(self):
        """Test log_base_prompt_success method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        base_prompt = "A beautiful sci-fi landscape with futuristic buildings and neon lights"
        progress_logger.log_base_prompt_success(base_prompt, 2.5)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "Base prompt generated" in call_args
        assert "(2.50s)" in call_args
        assert "A beautiful sci-fi landscape" in call_args

    def test_log_base_prompt_success_without_time(self):
        """Test log_base_prompt_success method without generation time."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        base_prompt = "A beautiful sci-fi landscape"
        progress_logger.log_base_prompt_success(base_prompt)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "Base prompt generated" in call_args
        assert "(" not in call_args  # No time info
        assert "A beautiful sci-fi landscape" in call_args

    def test_log_base_prompt_success_long_prompt(self):
        """Test log_base_prompt_success method with long prompt."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        base_prompt = "A" * 100  # 100 character prompt
        progress_logger.log_base_prompt_success(base_prompt, 1.0)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "..." in call_args  # Should be truncated
        # Check that the prompt was truncated (should contain "..." and be shorter than original)
        assert base_prompt[:80] + "..." in call_args

    def test_log_variant_prompt_generation_start(self):
        """Test log_variant_prompt_generation_start method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.log_variant_prompt_generation_start(5, "openai", "gpt-5", True)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "🔀" in call_args
        assert "5 prompt variants" in call_args
        assert "openai/gpt-5" in call_args
        assert "parallel mode" in call_args

    def test_log_variant_prompt_generation_start_sequential(self):
        """Test log_variant_prompt_generation_start method in sequential mode."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.log_variant_prompt_generation_start(3, "dummy", "local", False)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "🔀" in call_args
        assert "3 prompt variants" in call_args
        assert "dummy/local" in call_args
        assert "sequential mode" in call_args

    def test_log_variant_prompt_success(self):
        """Test log_variant_prompt_success method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        variant_prompts = [
            "Variant 1: A sci-fi scene with robots",
            "Variant 2: A fantasy landscape with dragons",
            "Variant 3: A cyberpunk city at night"
        ]
        progress_logger.log_variant_prompt_success(variant_prompts, 5.2)
        
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()
        
        # Check main success message
        info_call_args = mock_logger.info.call_args[0][0]
        assert "✓" in info_call_args
        assert "Generated 3 prompt variants" in info_call_args
        assert "(5.20s)" in info_call_args
        
        # Check that debug was called for each variant
        assert mock_logger.debug.call_count == 3
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert "#1" in debug_calls[0]
        assert "#2" in debug_calls[1]
        assert "#3" in debug_calls[2]

    def test_log_variant_prompt_success_without_time(self):
        """Test log_variant_prompt_success method without generation time."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        variant_prompts = ["Variant 1", "Variant 2"]
        progress_logger.log_variant_prompt_success(variant_prompts)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "✓" in call_args
        assert "Generated 2 prompt variants" in call_args
        assert "(" not in call_args  # No time info

    def test_log_variant_prompt_success_long_variants(self):
        """Test log_variant_prompt_success method with long variant prompts."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        long_variant = "A" * 80  # 80 character variant
        variant_prompts = [long_variant]
        progress_logger.log_variant_prompt_success(variant_prompts, 1.0)
        
        mock_logger.debug.assert_called()
        debug_call_args = mock_logger.debug.call_args[0][0]
        assert "..." in debug_call_args  # Should be truncated
        assert "#1" in debug_call_args

    def test_log_variant_prompt_error(self):
        """Test log_variant_prompt_error method."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.log_variant_prompt_error("API rate limit exceeded", 3.1)
        
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "✗" in call_args
        assert "Variant prompt generation failed" in call_args
        assert "(after 3.10s)" in call_args
        assert "API rate limit exceeded" in call_args

    def test_log_variant_prompt_error_without_time(self):
        """Test log_variant_prompt_error method without generation time."""
        mock_logger = Mock()
        progress_logger = ProgressLogger(mock_logger)
        
        progress_logger.log_variant_prompt_error("Connection timeout")
        
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "✗" in call_args
        assert "Variant prompt generation failed" in call_args
        assert "Connection timeout" in call_args
        assert "(" not in call_args  # No time info
