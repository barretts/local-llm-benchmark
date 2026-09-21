from decimal import Decimal, ROUND_HALF_UP

def _decimal(value, minimum, maximum=None):
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("finite Decimal required")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("out of range")

def settle(lines, tax_rate) -> Decimal:
    _decimal(tax_rate, Decimal(0), Decimal(1))
    cent = Decimal("0.01")
    subtotal = Decimal("0.00")
    for price, quantity, discount in lines:
        _decimal(price, Decimal(0))
        _decimal(discount, Decimal(0), Decimal(1))
        if type(quantity) is not int or quantity < 0:
            raise ValueError("invalid quantity")
        subtotal += (price * quantity * (Decimal(1) - discount)).quantize(cent, rounding=ROUND_HALF_UP)
    tax = (subtotal * tax_rate).quantize(cent, rounding=ROUND_HALF_UP)
    return subtotal + tax
