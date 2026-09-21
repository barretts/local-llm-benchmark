from decimal import Decimal, ROUND_HALF_EVEN

CENT = Decimal("0.01")

def _decimal(value, minimum, maximum=None):
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("finite Decimal required")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("out of range")

def line_total(unit_price, quantity, discount) -> Decimal:
    _decimal(unit_price, Decimal(0))
    _decimal(discount, Decimal(0), Decimal(1))
    if type(quantity) is not int or quantity < 0:
        raise ValueError("invalid quantity")
    return (unit_price * quantity * (Decimal(1) - discount)).quantize(CENT, rounding=ROUND_HALF_EVEN)
