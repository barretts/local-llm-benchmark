import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.models import LineItem
from src.money import line_total
from src.invoice import invoice_total
from dataclasses import FrozenInstanceError
def total(lines, rate):
    return invoice_total([LineItem(Decimal(p), q, Decimal(d)) for p, q, d in lines], Decimal(rate))
def RAW_TOTAL(lines, rate):
    return invoice_total([LineItem(p, q, d) for p, q, d in lines], rate)
CASES = [{'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['144.05', 4, '0'], ['61.947', 9, '0.1'], ['318.119', 4, '0.33']], 'tax_rate': '0.005', 'line_totals': ['0.01', '0.01', '576.20', '501.77', '852.56'], 'total': '1940.20'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['298.312', 9, '0.5'], ['18.153', 3, '0.33']], 'tax_rate': '0.2', 'line_totals': ['0.01', '0.01', '1342.40', '36.49'], 'total': '1654.69'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['97.347', 12, '0.1'], ['2', 8, '0.05']], 'tax_rate': '0.5', 'line_totals': ['0.01', '0.01', '1051.35', '15.20'], 'total': '1599.86'}, {'lines': [['175.933', 4, '0.25'], ['173.863', 11, '0.33'], ['178.555', 3, '0'], ['468.23', 1, '0.1']], 'tax_rate': '1', 'line_totals': ['527.80', '1281.37', '535.67', '421.41'], 'total': '5532.50'}, {'lines': [['292.532', 8, '0.33'], ['323.274', 3, '0.5'], ['130.999', 4, '0'], ['300.54', 1, '0.5']], 'tax_rate': '0.5', 'line_totals': ['1567.97', '484.91', '524.00', '150.27'], 'total': '4090.73'}, {'lines': [['151.012', 4, '0.5']], 'tax_rate': '0.005', 'line_totals': ['302.02'], 'total': '303.53'}, {'lines': [['59.148', 2, '0.5'], ['16.394', 11, '0.25']], 'tax_rate': '0.005', 'line_totals': ['59.15', '135.25'], 'total': '195.37'}, {'lines': [['470.127', 6, '0'], ['129.54', 6, '0.1'], ['10.809', 11, '0'], ['304.659', 6, '0'], ['270.132', 9, '1'], ['73.724', 5, '0.33'], ['221.937', 2, '0.33'], ['166.313', 11, '0']], 'tax_rate': '0.01', 'line_totals': ['2820.76', '699.52', '118.90', '1827.95', '0.00', '246.98', '297.40', '1829.44'], 'total': '7919.36'}, {'lines': [['280.556', 12, '0.05'], ['331.003', 1, '0'], ['450.793', 0, '0.1'], ['488.767', 6, '0.5']], 'tax_rate': '0.01', 'line_totals': ['3198.34', '331.00', '0.00', '1466.30'], 'total': '5045.60'}, {'lines': [], 'tax_rate': '1', 'line_totals': [], 'total': '0.00'}, {'lines': [['67.1', 8, '0.05'], ['27.37', 7, '1'], ['368.032', 11, '0.5'], ['340.415', 6, '1'], ['369.733', 0, '0.5'], ['309.374', 12, '0']], 'tax_rate': '1', 'line_totals': ['509.96', '0.00', '2024.18', '0.00', '0.00', '3712.49'], 'total': '12493.26'}, {'lines': [['40.714', 4, '0.05'], ['81.342', 0, '0.1']], 'tax_rate': '0', 'line_totals': ['154.71', '0.00'], 'total': '154.71'}, {'lines': [['358.24', 9, '0.1'], ['246.568', 11, '0.5'], ['480.864', 1, '0.25'], ['260.324', 3, '0.05'], ['48.314', 9, '0.33']], 'tax_rate': '0.125', 'line_totals': ['2901.74', '1356.12', '360.65', '741.92', '291.33'], 'total': '6358.23'}, {'lines': [['347.087', 5, '1'], ['44.302', 8, '0'], ['320.783', 8, '0.5'], ['171.218', 6, '0.25']], 'tax_rate': '0.01', 'line_totals': ['0.00', '354.42', '1283.13', '770.48'], 'total': '2432.11'}, {'lines': [['167.71', 2, '0.33'], ['341.836', 2, '0.1']], 'tax_rate': '1', 'line_totals': ['224.73', '615.30'], 'total': '1680.06'}, {'lines': [['191.789', 10, '0'], ['90.666', 1, '0.25'], ['485.318', 5, '1']], 'tax_rate': '0.005', 'line_totals': ['1917.89', '68.00', '0.00'], 'total': '1995.82'}, {'lines': [['332.303', 10, '0.25'], ['465.394', 12, '0.05'], ['249.676', 6, '0.5'], ['320.274', 6, '0.33'], ['15.13', 6, '0.25']], 'tax_rate': '1', 'line_totals': ['2492.27', '5305.49', '749.03', '1287.50', '68.09'], 'total': '19804.76'}, {'lines': [['119.415', 3, '0.25']], 'tax_rate': '0.125', 'line_totals': ['268.68'], 'total': '302.27'}, {'lines': [['272.135', 1, '0.5'], ['476.083', 2, '0.1'], ['419.27', 5, '0.25'], ['216.93', 1, '0'], ['185.427', 1, '0.5'], ['27.879', 5, '0.5']], 'tax_rate': '0.5', 'line_totals': ['136.07', '856.95', '1572.26', '216.93', '92.71', '69.70'], 'total': '4416.93'}, {'lines': [['391.289', 5, '0.5'], ['284.685', 6, '0.25']], 'tax_rate': '0.01', 'line_totals': ['978.22', '1281.08'], 'total': '2281.89'}, {'lines': [], 'tax_rate': '0.005', 'line_totals': [], 'total': '0.00'}, {'lines': [], 'tax_rate': '0.5', 'line_totals': [], 'total': '0.00'}, {'lines': [['499.937', 6, '0'], ['147.925', 3, '0.33'], ['21.969', 4, '0.5'], ['332.807', 4, '0'], ['270.864', 3, '0.5']], 'tax_rate': '0.005', 'line_totals': ['2999.62', '297.33', '43.94', '1331.23', '406.30'], 'total': '5103.81'}, {'lines': [['208.878', 6, '0.1']], 'tax_rate': '0.125', 'line_totals': ['1127.94'], 'total': '1268.93'}, {'lines': [], 'tax_rate': '0', 'line_totals': [], 'total': '0.00'}, {'lines': [['98.668', 0, '0.5'], ['388.481', 8, '0.5'], ['154.255', 2, '0.25'], ['379.09', 5, '0.33'], ['11.437', 7, '1'], ['128.996', 3, '0.1'], ['425.008', 4, '0.1']], 'tax_rate': '0', 'line_totals': ['0.00', '1553.92', '231.38', '1269.95', '0.00', '348.29', '1530.03'], 'total': '4933.57'}, {'lines': [['303.58', 1, '0.25'], ['240.511', 6, '0.33'], ['403.58', 6, '0.25'], ['336.849', 6, '0'], ['179.02', 8, '0.33'], ['6.451', 0, '0.25'], ['331.544', 1, '1']], 'tax_rate': '0.005', 'line_totals': ['227.69', '966.85', '1816.11', '2021.09', '959.55', '0.00', '0.00'], 'total': '6021.25'}, {'lines': [['398.563', 7, '1'], ['321.145', 0, '0.1'], ['293.16', 1, '0.05']], 'tax_rate': '0', 'line_totals': ['0.00', '0.00', '278.50'], 'total': '278.50'}, {'lines': [['435.685', 6, '0.33'], ['120.16', 2, '0'], ['433.905', 2, '0.25'], ['140.7', 2, '0.25'], ['420.429', 6, '1']], 'tax_rate': '0.01', 'line_totals': ['1751.45', '240.32', '650.86', '211.05', '0.00'], 'total': '2882.22'}, {'lines': [['160.576', 0, '0.1'], ['304.497', 3, '0.25'], ['33.98', 10, '0.05'], ['227.171', 6, '0'], ['254.14', 3, '0.33'], ['14.82', 11, '0.25'], ['204.73', 7, '0.1']], 'tax_rate': '0', 'line_totals': ['0.00', '685.12', '322.81', '1363.03', '510.82', '122.27', '1289.80'], 'total': '4293.85'}, {'lines': [['425.675', 1, '0.5'], ['282.115', 12, '0.5'], ['498.6', 4, '0'], ['280.987', 5, '0.1'], ['191.087', 2, '1'], ['342.656', 11, '1']], 'tax_rate': '0', 'line_totals': ['212.84', '1692.69', '1994.40', '1264.44', '0.00', '0.00'], 'total': '5164.37'}, {'lines': [['286.052', 0, '0.1']], 'tax_rate': '1', 'line_totals': ['0.00'], 'total': '0.00'}, {'lines': [['454.429', 12, '0'], ['63.229', 10, '0.33'], ['285.405', 1, '0'], ['299.032', 4, '0.05'], ['11.651', 9, '0.33'], ['465.937', 2, '0.5'], ['280.717', 8, '0.05'], ['323.57', 5, '0.1']], 'tax_rate': '0.125', 'line_totals': ['5453.15', '423.63', '285.41', '1136.32', '70.26', '465.94', '2133.45', '1456.07'], 'total': '12852.26'}, {'lines': [['15.042', 4, '0.1'], ['176.172', 8, '0.33'], ['10.203', 5, '0']], 'tax_rate': '0.2', 'line_totals': ['54.15', '944.28', '51.02'], 'total': '1259.34'}, {'lines': [], 'tax_rate': '1', 'line_totals': [], 'total': '0.00'}, {'lines': [['266.453', 1, '0.1'], ['117.717', 11, '0.33'], ['265.67', 0, '0.5'], ['434.929', 6, '0'], ['38.276', 4, '0.05']], 'tax_rate': '0.125', 'line_totals': ['239.81', '867.57', '0.00', '2609.57', '145.45'], 'total': '4345.20'}, {'lines': [['129.772', 2, '1'], ['28.905', 10, '0.5'], ['499.868', 8, '1']], 'tax_rate': '0.005', 'line_totals': ['0.00', '144.53', '0.00'], 'total': '145.25'}, {'lines': [['393.434', 10, '0.5'], ['22.86', 4, '0']], 'tax_rate': '0.005', 'line_totals': ['1967.17', '91.44'], 'total': '2068.90'}, {'lines': [], 'tax_rate': '0.5', 'line_totals': [], 'total': '0.00'}, {'lines': [['8.506', 1, '0.1'], ['315.766', 4, '0.05'], ['451.31', 6, '0.5'], ['367.768', 2, '0'], ['260.943', 10, '1'], ['400.037', 0, '1'], ['477.826', 3, '0.33']], 'tax_rate': '0.5', 'line_totals': ['7.66', '1199.91', '1353.93', '735.54', '0.00', '0.00', '960.43'], 'total': '6386.21'}, {'lines': [['476.861', 5, '0.5']], 'tax_rate': '1', 'line_totals': ['1192.15'], 'total': '2384.30'}, {'lines': [], 'tax_rate': '0.5', 'line_totals': [], 'total': '0.00'}, {'lines': [['232.62', 1, '1'], ['361.342', 11, '0.25']], 'tax_rate': '0.125', 'line_totals': ['0.00', '2981.07'], 'total': '3353.70'}, {'lines': [['383.158', 1, '0.1'], ['17.742', 2, '0.05'], ['431.349', 11, '0.25'], ['440.521', 4, '0.1']], 'tax_rate': '0', 'line_totals': ['344.84', '33.71', '3558.63', '1585.88'], 'total': '5523.06'}, {'lines': [['285.402', 4, '0']], 'tax_rate': '0.005', 'line_totals': ['1141.61'], 'total': '1147.32'}]

class Hidden(unittest.TestCase):
    def test_tax_half_cent_separate_rounding(self):
        self.assertEqual(total([["0.05", 1, "0"]], "0.1"), Decimal("0.06"))
        self.assertEqual(total([["0.005", 1, "0"]], "0.5"), Decimal("0.02"))
    def test_quantity_discount_and_zero(self):
        self.assertEqual(total([["1.005", 3, "0.5"]], "0"), Decimal("1.51"))
        self.assertEqual(total([["999.99", 0, "0.2"], ["12.34", 7, "1"]], "0.9"), Decimal("0.00"))
    def test_empty_is_two_decimal_zero(self):
        result = total([], "0.9")
        self.assertIsInstance(result, Decimal)
        self.assertEqual(str(result), "0.00")
    def test_invalid_prices_and_discounts(self):
        for price in (Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity"), 1.2):
            with self.subTest(price=price), self.assertRaises(ValueError):
                RAW_TOTAL([(price, 1, Decimal(0))], Decimal(0))
        for discount in (Decimal("-0.01"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity"), 0.5):
            with self.subTest(discount=discount), self.assertRaises(ValueError):
                RAW_TOTAL([(Decimal(1), 1, discount)], Decimal(0))
    def test_invalid_quantities_and_tax(self):
        for quantity in (-1, 1.5, True, "1"):
            with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                RAW_TOTAL([(Decimal(1), quantity, Decimal(0))], Decimal(0))
        for rate in (Decimal("-0.01"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity"), 0.5):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                RAW_TOTAL([], rate)
    def test_input_unchanged_and_decimal_precision(self):
        lines = [(Decimal("123456789.125"), 3, Decimal("0.125")), (Decimal("0.005"), 1, Decimal(0))]
        before = copy.deepcopy(lines)
        result = RAW_TOTAL(lines, Decimal("0.125"))
        subtotal = Decimal("324074071.45") + Decimal("0.01")
        expected = subtotal + (subtotal * Decimal("0.125")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        self.assertEqual(result, expected)
        self.assertEqual(lines, before)
    def test_seeded_decimal_invoices(self):
        for index, case in enumerate(CASES):
            with self.subTest(index=index):
                result = total(case["lines"], case["tax_rate"])
                self.assertIsInstance(result, Decimal)
                self.assertEqual(result, Decimal(case["total"]))
                self.assertEqual(str(result), case["total"])

    def test_individual_helper_and_integrated_consistency(self):
        immutable = LineItem(Decimal("1.00"), 1, Decimal("0"))
        with self.assertRaises(FrozenInstanceError):
            immutable.unit_price = Decimal("2.00")
        for index, case in enumerate(CASES):
            with self.subTest(index=index):
                lines = [LineItem(Decimal(p), q, Decimal(d)) for p, q, d in case["lines"]]
                individual = [line_total(item.unit_price, item.quantity, item.discount) for item in lines]
                self.assertEqual(individual, [Decimal(v) for v in case["line_totals"]])
                subtotal = sum(individual, Decimal("0.00"))
                expected = subtotal + (subtotal * Decimal(case["tax_rate"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                self.assertEqual(invoice_total(lines, Decimal(case["tax_rate"])), expected)


if __name__ == '__main__':
    unittest.main()
