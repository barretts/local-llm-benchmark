from decimal import Decimal, ROUND_HALF_UP
from .models import LineItem
from .money import line_total, _decimal

def invoice_total(items, tax_rate) -> Decimal:
    _decimal(tax_rate, Decimal(0), Decimal(1))
    subtotal = sum((line_total(item.unit_price, item.quantity, item.discount) for item in items), Decimal("0.00"))
    tax = (subtotal * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return subtotal + tax
