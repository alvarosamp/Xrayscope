from fluent import handler as fluent_handler
import logging

class ExtraFieldsFluentFormatter(fluent_handler.FluentRecordFormatter):
    """
    Custom Fluentd formatter that automatically includes all extra fields from log records
    """
    def format(self, record):
        data = super().format(record)
        
        # List of default LogRecord attributes to exclude
        default_fields = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename', 
            'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs', 
            'message', 'msg', 'name', 'pathname', 'process', 'processName', 
            'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
        
        # Add all non-default fields (extra fields)
        for key in record.__dict__:
            if key not in default_fields and key not in data:
                data[key] = getattr(record, key)
        
        return data

def configure_fluent_logging(logger_name: str, service_name: str, fluent_host: str, fluent_port: int):
    """
    Configure fluent logging with our custom formatter
    Returns the logger instance for immediate use
    """
    logger = logging.getLogger(logger_name)
    
    # Base format configuration
    base_format = {
        'service': service_name,
        'where': '%(module)s.%(funcName)s',
        'level': '%(levelname)s',
        'message': '%(message)s'
    }
    
    # Create and configure handler (fixed variable name)
    handler = fluent_handler.FluentHandler('model', host=fluent_host, port=fluent_port)
    formatter = ExtraFieldsFluentFormatter(base_format)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger