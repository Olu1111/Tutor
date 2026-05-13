from __future__ import annotations

from dataclasses import dataclass, field
import ast
import contextlib
import io
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
}

SAFE_METHODS_BY_TYPE = {
    "list": {"append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove", "reverse", "sort"},
    "tuple": {"count", "index"},
    "dict": {"clear", "copy", "get", "items", "keys", "pop", "popitem", "setdefault", "update", "values"},
    "set": {"add", "clear", "copy", "difference", "discard", "intersection", "isdisjoint", "issubset", "issuperset", "pop", "remove", "union"},
}

SAFE_METHOD_NAMES: Set[str] = set().union(*SAFE_METHODS_BY_TYPE.values())

FORBIDDEN_NAMES = {
    "__import__",
    "__builtins__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "help",
    "breakpoint",
}

FORBIDDEN_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.Lambda,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Try,
    ast.Raise,
)


@dataclass(frozen=True)
class EvaluationCase:
    args: Tuple[Any, ...]
    expected: Any = None
    expected_stdout: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuizExercise:
    name: str
    prompt: str
    function_name: str
    cases: Sequence[EvaluationCase]
    starter_code: str


@dataclass
class CaseResult:
    case_index: int
    passed: bool
    actual: str = ""
    expected: str = ""
    stdout: str = ""
    error: str = ""


@dataclass
class EvaluationResult:
    passed_count: int
    total_count: int
    blocked_reason: str = ""
    timed_out: bool = False
    syntax_error: str = ""
    case_results: List[CaseResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.total_count > 0 and self.passed_count == self.total_count and not self.blocked_reason and not self.timed_out and not self.syntax_error

    def summary(self) -> str:
        if self.blocked_reason:
            return f"Blocked: {self.blocked_reason}"
        if self.syntax_error:
            return f"Syntax error: {self.syntax_error}"
        if self.timed_out:
            return "Timed out while running tests."
        return f"{self.passed_count}/{self.total_count} tests passed."


DEFAULT_EXERCISES: Sequence[QuizExercise] = (
    QuizExercise(
        name="Factorial",
        prompt="Implement factorial(n) so it returns n! for non-negative integers.",
        function_name="factorial",
        cases=(
            EvaluationCase(args=(0,), expected=1),
            EvaluationCase(args=(5,), expected=120),
            EvaluationCase(args=(6,), expected=720),
        ),
        starter_code="""def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n""",
    ),
    QuizExercise(
        name="Sum to n",
        prompt="Implement sum_to_n(n) so it returns 1 + 2 + ... + n.",
        function_name="sum_to_n",
        cases=(
            EvaluationCase(args=(0,), expected=0),
            EvaluationCase(args=(5,), expected=15),
            EvaluationCase(args=(10,), expected=55),
        ),
        starter_code="""def sum_to_n(n):\n    if n == 0:\n        return 0\n    return n + sum_to_n(n - 1)\n""",
    ),
    QuizExercise(
        name="Fibonacci",
        prompt="Implement fib(n) using recursion or iteration.",
        function_name="fib",
        cases=(
            EvaluationCase(args=(0,), expected=0),
            EvaluationCase(args=(1,), expected=1),
            EvaluationCase(args=(8,), expected=21),
        ),
        starter_code="""def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n""",
    ),
    QuizExercise(
        name="Reverse string",
        prompt="Implement reverse_string(s) so it returns the string backwards.",
        function_name="reverse_string",
        cases=(
            EvaluationCase(args=("",), expected=""),
            EvaluationCase(args=("abc",), expected="cba"),
            EvaluationCase(args=("Hello",), expected="olleH"),
        ),
        starter_code="""def reverse_string(s):\n    if s == "":\n        return s\n    return reverse_string(s[1:]) + s[0]\n""",
    ),
    QuizExercise(
        name="Count down",
        prompt="Implement count_down(n) so it prints numbers from n down to 1.",
        function_name="count_down",
        cases=(
            EvaluationCase(args=(3,), expected=None, expected_stdout="3\n2\n1\n"),
            EvaluationCase(args=(1,), expected=None, expected_stdout="1\n"),
        ),
        starter_code="""def count_down(n):\n    if n <= 0:\n        return\n    print(n)\n    count_down(n - 1)\n""",
    ),
)


def get_default_exercises() -> Sequence[QuizExercise]:
    return DEFAULT_EXERCISES


def load_source_from_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _build_parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_allowed_method_call(node: ast.Attribute, parents: Dict[ast.AST, ast.AST]) -> bool:
    if node.attr.startswith("__") or node.attr not in SAFE_METHOD_NAMES:
        return False

    parent = parents.get(node)
    if not isinstance(parent, ast.Call) or parent.func is not node:
        return False

    base = node.value
    if isinstance(base, ast.Name):
        return base.id not in FORBIDDEN_NAMES

    return isinstance(base, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Subscript, ast.Call))


def _validate_ast(tree: ast.AST) -> None:
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise ValueError(f"{type(node).__name__} is not allowed in the evaluator.")
        if isinstance(node, ast.Attribute):
            if not _is_allowed_method_call(node, parents):
                raise ValueError("Only a limited set of safe collection method calls are allowed in the evaluator.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Call):
            call_target = node.func
            if isinstance(call_target, ast.Name) and call_target.id in FORBIDDEN_NAMES:
                raise ValueError(f"Call to '{call_target.id}' is not allowed.")
            if isinstance(call_target, ast.Attribute):
                if not _is_allowed_method_call(call_target, parents):
                    raise ValueError("Only safe collection method calls are allowed in the evaluator.")


def _parse_and_validate(source: str) -> ast.AST:
    if len(source) > 20_000:
        raise ValueError("Code submission is too large for safe evaluation.")

    tree = ast.parse(source, mode="exec")
    _validate_ast(tree)
    return tree


def _run_case(function: Callable[..., Any], case: EvaluationCase) -> Tuple[Any, str]:
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        value = function(*case.args, **case.kwargs)
    return value, stdout_buffer.getvalue()


def _worker(source: str, exercise: QuizExercise, queue: mp.Queue) -> None:
    try:
        tree = _parse_and_validate(source)
    except SyntaxError as error:
        queue.put({"syntax_error": f"{error.msg} (line {error.lineno})"})
        return
    except Exception as error:
        queue.put({"blocked_reason": str(error)})
        return

    sandbox_globals: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__student__",
    }

    try:
        exec(compile(tree, "<student_submission>", "exec"), sandbox_globals, sandbox_globals)
    except Exception as error:
        queue.put({"blocked_reason": f"Error while loading code: {error}"})
        return

    student_function = sandbox_globals.get(exercise.function_name)
    if not callable(student_function):
        queue.put({"blocked_reason": f"Function '{exercise.function_name}' was not defined."})
        return

    results: List[CaseResult] = []
    passed_count = 0
    for index, case in enumerate(exercise.cases, start=1):
        try:
            actual, stdout = _run_case(student_function, case)
            passed = actual == case.expected and stdout == case.expected_stdout
            if passed:
                passed_count += 1
            results.append(
                CaseResult(
                    case_index=index,
                    passed=passed,
                    actual=repr(actual),
                    expected=repr(case.expected),
                    stdout=repr(stdout),
                    error="",
                )
            )
        except Exception as error:
            results.append(
                CaseResult(
                    case_index=index,
                    passed=False,
                    actual="",
                    expected=repr(case.expected),
                    stdout="",
                    error=str(error),
                )
            )

    queue.put(
        {
            "passed_count": passed_count,
            "total_count": len(exercise.cases),
            "case_results": results,
        }
    )


def evaluate_submission(source: str, exercise: QuizExercise, timeout_seconds: int = 3) -> EvaluationResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    process = context.Process(target=_worker, args=(source, exercise, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return EvaluationResult(passed_count=0, total_count=len(exercise.cases), timed_out=True)

    if queue.empty():
        return EvaluationResult(
            passed_count=0,
            total_count=len(exercise.cases),
            blocked_reason="No evaluation result was returned.",
        )

    payload = queue.get()
    if "syntax_error" in payload:
        return EvaluationResult(
            passed_count=0,
            total_count=len(exercise.cases),
            syntax_error=payload["syntax_error"],
        )
    if "blocked_reason" in payload:
        return EvaluationResult(
            passed_count=0,
            total_count=len(exercise.cases),
            blocked_reason=payload["blocked_reason"],
        )

    return EvaluationResult(
        passed_count=payload["passed_count"],
        total_count=payload["total_count"],
        case_results=payload["case_results"],
    )
