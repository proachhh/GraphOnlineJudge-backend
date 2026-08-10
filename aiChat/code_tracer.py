"""
Python 代码执行追踪器
使用 sys.settrace 捕获每一步的变量状态、调用栈和输出
"""
import sys
import io
import json
import traceback
import signal
from contextlib import redirect_stdout


def trace_code(code, input_data='', max_steps=800, timeout_sec=5):
    """执行 Python 代码并返回逐步执行轨迹"""
    steps = []
    output_buffer = io.StringIO()
    input_lines = input_data.split('\n') if input_data else []
    input_idx = [0]

    def custom_input(prompt=''):
        if input_idx[0] < len(input_lines):
            val = input_lines[input_idx[0]]
            input_idx[0] += 1
            print(prompt + val)
            return val
        return ''

    def safe_str(obj, max_len=200):
        try:
            s = repr(obj)
            if len(s) > max_len:
                s = s[:max_len] + '...'
            return s
        except Exception:
            return '<unreprable>'

    def serialize_vars(var_dict, max_vars=20):
        result = {}
        count = 0
        for k, v in var_dict.items():
            if k.startswith('__'):
                continue
            if type(v).__name__ == 'module':
                continue
            if count >= max_vars:
                break
            try:
                result[k] = {
                    'value': safe_str(v),
                    'type': type(v).__name__
                }
                count += 1
            except Exception:
                pass
        return result

    def get_call_stack(frame):
        stack = []
        f = frame
        depth = 0
        while f is not None and depth < 10:
            name = f.f_code.co_name
            if name == '<module>':
                break
            stack.append({
                'function': name,
                'line': f.f_lineno,
                'filename': f.f_code.co_filename.replace('<string>', 'main')
            })
            f = f.f_back
            depth += 1
        return list(reversed(stack))

    line_count = [0]

    def trace_func(frame, event, arg):
        if event == 'line':
            if line_count[0] >= max_steps:
                return None
            line_count[0] += 1

            steps.append({
                'type': 'line',
                'line': frame.f_lineno,
                'locals': serialize_vars(frame.f_locals),
                'globals': {} if frame.f_back is None else {},
                'stack': get_call_stack(frame),
                'depth': len(get_call_stack(frame)),
                'output': output_buffer.getvalue()
            })
        elif event == 'call':
            func_name = frame.f_code.co_name
            if func_name not in ('_find_and_load', '_find_spec', '_gcd_import',
                                  '__import__', '_handle_fromlist', '_call_with_frames_cleaned'):
                call_stack = get_call_stack(frame)
                steps.append({
                    'type': 'call',
                    'line': frame.f_lineno,
                    'function': func_name,
                    'locals': serialize_vars(frame.f_locals),
                    'stack': call_stack,
                    'depth': len(call_stack),
                    'output': output_buffer.getvalue()
                })
        elif event == 'return':
            func_name = frame.f_code.co_name
            if func_name not in ('_find_and_load', '_find_spec', '_gcd_import'):
                call_stack = get_call_stack(frame)
                steps.append({
                    'type': 'return',
                    'line': frame.f_lineno,
                    'function': func_name,
                    'return_value': safe_str(arg) if arg is not None else 'None',
                    'locals': serialize_vars(frame.f_locals),
                    'stack': call_stack,
                    'depth': len(call_stack),
                    'output': output_buffer.getvalue()
                })
        elif event == 'exception':
            exc_type, exc_value, exc_tb = arg
            call_stack = get_call_stack(frame)
            steps.append({
                'type': 'exception',
                'line': frame.f_lineno,
                'exception': exc_type.__name__,
                'message': str(exc_value),
                'locals': serialize_vars(frame.f_locals),
                'stack': call_stack,
                'depth': len(call_stack),
                'output': output_buffer.getvalue()
            })
        return trace_func

    # 限制可用的内置函数
    safe_builtins = {
        'print': print,
        'len': len, 'range': range, 'int': int, 'float': float,
        'str': str, 'bool': bool, 'list': list, 'dict': dict,
        'set': set, 'tuple': tuple, 'abs': abs, 'min': min, 'max': max,
        'sum': sum, 'sorted': sorted, 'reversed': reversed,
        'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
        'any': any, 'all': all, 'round': round, 'pow': pow,
        'isinstance': isinstance, 'type': type, 'ord': ord, 'chr': chr,
        'hex': hex, 'bin': bin, 'oct': oct, 'format': format,
        'input': custom_input, 'split': str.split,
        'open': open, 'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'IndexError': IndexError,
        'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError,
        'StopIteration': StopIteration, 'NameError': NameError,
        'AttributeError': AttributeError, 'NotImplementedError': NotImplementedError,
        'RuntimeError': RuntimeError, 'ImportError': ImportError,
        '__build_class__': __build_class__, '__name__': '__main__',
    }

    # 可用模块
    import math
    import random
    import collections
    import itertools
    import functools
    safe_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        'math': math,
        'random': random,
        'collections': collections,
        'itertools': itertools,
        'functools': functools,
    }

    def timeout_handler(signum, frame):
        raise TimeoutError("执行超时")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)

        with redirect_stdout(output_buffer):
            compiled = compile(code, '<string>', 'exec')
            sys.settrace(trace_func)
            exec(compiled, safe_globals)
            sys.settrace(None)

        signal.alarm(0)

        steps.append({
            'type': 'end',
            'line': -1,
            'output': output_buffer.getvalue(),
            'locals': {}
        })
    except TimeoutError:
        sys.settrace(None)
        signal.alarm(0)
        steps.append({
            'type': 'error',
            'line': -1,
            'error': 'TimeoutError',
            'message': f'执行超过 {timeout_sec} 秒限制',
            'output': output_buffer.getvalue()
        })
    except Exception as e:
        sys.settrace(None)
        signal.alarm(0)
        tb = traceback.format_exc()
        # 提取错误行号
        error_line = -1
        for line in tb.split('\n'):
            if 'line' in line and '<string>' in line:
                try:
                    error_line = int(line.strip().split('line')[1].split(',')[0].strip())
                except (IndexError, ValueError):
                    pass
        steps.append({
            'type': 'error',
            'line': error_line,
            'error': type(e).__name__,
            'message': str(e),
            'traceback': tb,
            'output': output_buffer.getvalue()
        })

    return steps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', required=True, help='Python code to trace')
    parser.add_argument('--input', default='', help='Standard input')
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--timeout', type=int, default=5)
    args = parser.parse_args()

    result = trace_code(args.code, args.input, args.max_steps, args.timeout)
    print(json.dumps(result, ensure_ascii=False))
