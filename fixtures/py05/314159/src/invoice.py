from decimal import Decimal, ROUND_HALF_UP
from .models import LineItem
from .money import line_total, _decimal

def invoice_total(items, tax_rate) -> Decimal:
    _decimal(tax_rate, Decimal(0), Decimal(1))
    subtotal = Decimal("0.00")
    for item in items:
        line_total(item.unit_price, item.quantity, item.discount)
        subtotal += item.unit_price * item.quantity * (Decimal(1) - item.discount)
    return (subtotal * (Decimal(1) + tax_rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
