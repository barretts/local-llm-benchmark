from decimal import Decimal

def _decimal(value, minimum, maximum=None):
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("finite Decimal required")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("out of range")

def settle(lines, tax_rate) -> Decimal:
    _decimal(tax_rate, Decimal(0), Decimal(1))
    subtotal = 0.0
    for price, quantity, discount in lines:
        _decimal(price, Decimal(0))
        _decimal(discount, Decimal(0), Decimal(1))
        if type(quantity) is not int or quantity < 0:
            raise ValueError("invalid quantity")
        subtotal += float(price) * quantity * (1.0 - float(discount))
    return Decimal(str(round(subtotal * (1.0 + float(tax_rate)), 2)))
