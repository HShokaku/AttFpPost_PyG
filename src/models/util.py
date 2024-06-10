def skip_computation_on_oom(return_value=None, error_message=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if error_message is not None:
                        print(error_message)
                    return return_value
                else:
                    raise e

        return wrapper

    return decorator
