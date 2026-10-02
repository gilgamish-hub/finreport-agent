import pytest

from finagent.grounding import extract_numbers, ungrounded_numbers
from finagent.tools import calculator, safe_eval


@pytest.mark.parametrize("expr, expected", [
    ("1577 - 1373", 204),
    ("(1577 - 1373) / 1373 * 100", 14.857975236707939),
    ("$1,577 + 2,000", 3577),
    ("round(2/3, 2)", 0.67),
    ("-(370) + 410", 40),
    ("2^3", 8),
])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "[1,2]", "9**9999"])
def test_safe_eval_rejects_code(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_calculator_tool_reports_errors_instead_of_raising():
    assert calculator.invoke({"expression": "1/0"}) == "error: division by zero"
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"


def test_extract_numbers_handles_report_formats():
    assert extract_numbers("$5,363 and (370) and 14.9%") == [5363, -370, 14.9]


def test_grounding_accepts_rounding_and_unit_changes():
    evidence = "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)"
    assert ungrounded_numbers("Capex was $1,577 million in FY2018.", evidence, []) == []
    assert ungrounded_numbers("Capex was about $1.58 billion.", evidence, []) == []
    assert ungrounded_numbers("Capex grew 14.86%.", evidence, ["(1577-1373)/1373*100 = 14.858"]) == []


def test_grounding_flags_invented_numbers():
    evidence = "Net sales | 32,765 | 31,657"
    assert ungrounded_numbers("Net sales were $35,000 million.", evidence, []) == [35000]
    assert ungrounded_numbers("Capex was $2,000 million in 2018.", evidence, []) == [2000]
