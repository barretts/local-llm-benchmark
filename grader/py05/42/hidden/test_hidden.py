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
CASES = [{'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['13.112', 11, '0.1']], 'tax_rate': '0.005', 'line_totals': ['0.01', '0.01', '129.81'], 'total': '130.48'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['73.158', 11, '0'], ['354.785', 11, '0.33'], ['45.58', 9, '0.25']], 'tax_rate': '0', 'line_totals': ['0.01', '0.01', '804.74', '2614.77', '307.67'], 'total': '3727.20'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0']], 'tax_rate': '0', 'line_totals': ['0.01', '0.01'], 'total': '0.02'}, {'lines': [['121.981', 8, '0.33'], ['13.912', 8, '0.05'], ['375.4', 10, '0.5']], 'tax_rate': '0.2', 'line_totals': ['653.82', '105.73', '1877.00'], 'total': '3163.86'}, {'lines': [['115.574', 7, '0.33'], ['145.852', 12, '1'], ['3.407', 12, '1'], ['83.707', 11, '0.25'], ['178.389', 4, '0.05'], ['112.886', 12, '0.1']], 'tax_rate': '0', 'line_totals': ['542.04', '0.00', '0.00', '690.58', '677.88', '1219.17'], 'total': '3129.67'}, {'lines': [['199.191', 1, '0.1']], 'tax_rate': '1', 'line_totals': ['179.27'], 'total': '358.54'}, {'lines': [['316.526', 4, '1'], ['22.78', 11, '0.25'], ['281.137', 1, '0.25'], ['41.313', 8, '0.1'], ['434.846', 10, '0.33']], 'tax_rate': '1', 'line_totals': ['0.00', '187.94', '210.85', '297.45', '2913.47'], 'total': '7219.42'}, {'lines': [['302.698', 3, '0.5'], ['36.466', 0, '0.5'], ['119.484', 12, '0.1'], ['41.833', 3, '1'], ['52.953', 6, '0.1']], 'tax_rate': '0.125', 'line_totals': ['454.05', '0.00', '1290.43', '0.00', '285.95'], 'total': '2284.23'}, {'lines': [['85.277', 5, '0.1'], ['109.842', 10, '0.1'], ['367.955', 10, '0.5'], ['37.435', 9, '0.5'], ['89.725', 8, '0.5']], 'tax_rate': '0.005', 'line_totals': ['383.75', '988.58', '1839.78', '168.46', '358.90'], 'total': '3758.17'}, {'lines': [['242.357', 6, '0.1'], ['485.171', 10, '0.5']], 'tax_rate': '0.2', 'line_totals': ['1308.73', '2425.86'], 'total': '4481.51'}, {'lines': [['358.935', 5, '1'], ['402.817', 12, '0'], ['120.087', 0, '1']], 'tax_rate': '0.01', 'line_totals': ['0.00', '4833.80', '0.00'], 'total': '4882.14'}, {'lines': [['140.373', 1, '0.05'], ['478.746', 9, '0.5'], ['164.981', 3, '0.5'], ['261.74', 6, '0.5'], ['240.57', 2, '0.1'], ['73.206', 3, '0.5']], 'tax_rate': '0.2', 'line_totals': ['133.35', '2154.36', '247.47', '785.22', '433.03', '109.81'], 'total': '4635.89'}, {'lines': [['137.752', 11, '0.33'], ['224.622', 9, '0.25'], ['189.79', 3, '0.05'], ['267.138', 7, '0'], ['396.247', 0, '1'], ['57.487', 2, '0.5'], ['83.876', 12, '0.5'], ['221.333', 9, '0']], 'tax_rate': '0.125', 'line_totals': ['1015.23', '1516.20', '540.90', '1869.97', '0.00', '57.49', '503.26', '1992.00'], 'total': '8431.93'}, {'lines': [['312.417', 7, '0.33'], ['131.813', 8, '1'], ['494.105', 0, '0.5'], ['377.865', 1, '0.5'], ['463.883', 8, '1'], ['139.893', 12, '0.5']], 'tax_rate': '0.01', 'line_totals': ['1465.24', '0.00', '0.00', '188.93', '0.00', '839.36'], 'total': '2518.47'}, {'lines': [['153.878', 6, '0.05']], 'tax_rate': '0.125', 'line_totals': ['877.10'], 'total': '986.74'}, {'lines': [], 'tax_rate': '0.5', 'line_totals': [], 'total': '0.00'}, {'lines': [['262.451', 12, '0.05'], ['266.171', 1, '1'], ['327.837', 4, '1'], ['334.993', 8, '0.33']], 'tax_rate': '0.005', 'line_totals': ['2991.94', '0.00', '0.00', '1795.56'], 'total': '4811.44'}, {'lines': [['196.038', 12, '0.05'], ['282.789', 12, '0.33']], 'tax_rate': '0', 'line_totals': ['2234.83', '2273.62'], 'total': '4508.45'}, {'lines': [['256.17', 0, '0'], ['487.115', 5, '1'], ['422.982', 4, '0.05'], ['30.369', 3, '0.33'], ['496.474', 1, '0']], 'tax_rate': '0.5', 'line_totals': ['0.00', '0.00', '1607.33', '61.04', '496.47'], 'total': '3247.26'}, {'lines': [['427.831', 1, '1'], ['279.291', 12, '0.05'], ['67.314', 10, '0.25'], ['496.421', 8, '0.05'], ['138.966', 8, '1'], ['318.029', 6, '0.05'], ['487.018', 8, '1']], 'tax_rate': '0.5', 'line_totals': ['0.00', '3183.92', '504.86', '3772.80', '0.00', '1812.77', '0.00'], 'total': '13911.53'}, {'lines': [['373.79', 4, '0.25'], ['352.157', 10, '0.1'], ['229.69', 8, '0.25']], 'tax_rate': '0', 'line_totals': ['1121.37', '3169.41', '1378.14'], 'total': '5668.92'}, {'lines': [['117.806', 1, '0.1'], ['11.028', 9, '0.33'], ['120.646', 9, '0.05']], 'tax_rate': '0', 'line_totals': ['106.03', '66.50', '1031.52'], 'total': '1204.05'}, {'lines': [['371.112', 10, '0']], 'tax_rate': '0.005', 'line_totals': ['3711.12'], 'total': '3729.68'}, {'lines': [['474.7', 0, '1']], 'tax_rate': '0.01', 'line_totals': ['0.00'], 'total': '0.00'}, {'lines': [['269.565', 3, '0.1']], 'tax_rate': '0.5', 'line_totals': ['727.83'], 'total': '1091.75'}, {'lines': [['112.321', 8, '0.05'], ['379.245', 9, '0.33'], ['247.815', 3, '1'], ['247.974', 12, '0.25'], ['99.829', 1, '0'], ['345.496', 6, '0.1'], ['222.077', 6, '0.25']], 'tax_rate': '1', 'line_totals': ['853.64', '2286.85', '0.00', '2231.77', '99.83', '1865.68', '999.35'], 'total': '16674.24'}, {'lines': [], 'tax_rate': '0.5', 'line_totals': [], 'total': '0.00'}, {'lines': [['31.778', 6, '0.5']], 'tax_rate': '0.01', 'line_totals': ['95.33'], 'total': '96.28'}, {'lines': [['130.367', 3, '0.05']], 'tax_rate': '0.2', 'line_totals': ['371.55'], 'total': '445.86'}, {'lines': [['73.495', 6, '0.05'], ['146.037', 7, '0.05'], ['458.482', 1, '0.25'], ['423.636', 8, '0'], ['26.522', 10, '0.33'], ['438.319', 0, '0'], ['485.683', 12, '1']], 'tax_rate': '0.005', 'line_totals': ['418.92', '971.15', '343.86', '3389.09', '177.70', '0.00', '0.00'], 'total': '5327.22'}, {'lines': [['213.078', 7, '0.25'], ['112.065', 6, '0']], 'tax_rate': '0.005', 'line_totals': ['1118.66', '672.39'], 'total': '1800.01'}, {'lines': [['1.13', 6, '0.1'], ['485.762', 12, '1'], ['238.555', 4, '0.25'], ['365.214', 11, '1'], ['291.382', 10, '0.5'], ['255.155', 2, '0.05']], 'tax_rate': '0.01', 'line_totals': ['6.10', '0.00', '715.67', '0.00', '1456.91', '484.79'], 'total': '2690.10'}, {'lines': [['30.662', 9, '0.5'], ['284.266', 0, '0.5'], ['164.419', 0, '0']], 'tax_rate': '0.2', 'line_totals': ['137.98', '0.00', '0.00'], 'total': '165.58'}, {'lines': [['263.638', 8, '0.05'], ['29.821', 8, '0'], ['446.348', 2, '0'], ['311.969', 1, '0.5'], ['451.841', 3, '0.25'], ['62.855', 9, '0.05'], ['303.52', 9, '0']], 'tax_rate': '0.2', 'line_totals': ['2003.65', '238.57', '892.70', '155.98', '1016.64', '537.41', '2731.68'], 'total': '9091.96'}, {'lines': [['219.794', 10, '0.33']], 'tax_rate': '0.2', 'line_totals': ['1472.62'], 'total': '1767.14'}, {'lines': [['165.868', 4, '0.05'], ['351.129', 11, '0.1'], ['125.14', 4, '0.25'], ['68.617', 10, '0.5'], ['157.284', 7, '0.1'], ['487.073', 12, '0'], ['4.883', 7, '0.33'], ['295.17', 1, '0']], 'tax_rate': '0.2', 'line_totals': ['630.30', '3476.18', '375.42', '343.09', '990.89', '5844.88', '22.90', '295.17'], 'total': '14374.60'}, {'lines': [['265.229', 4, '0.05'], ['489.296', 5, '0'], ['460.99', 3, '0.1']], 'tax_rate': '0.01', 'line_totals': ['1007.87', '2446.48', '1244.67'], 'total': '4746.01'}, {'lines': [['229.734', 8, '0.5'], ['158.606', 9, '1']], 'tax_rate': '0.5', 'line_totals': ['918.94', '0.00'], 'total': '1378.41'}, {'lines': [['4.101', 10, '1'], ['290.771', 4, '0.5'], ['54.309', 2, '0.1'], ['60.517', 1, '0.5'], ['290.048', 2, '0.1'], ['147.721', 9, '0.05'], ['376.235', 5, '0.05'], ['360.446', 10, '1']], 'tax_rate': '0.01', 'line_totals': ['0.00', '581.54', '97.76', '30.26', '522.09', '1263.01', '1787.12', '0.00'], 'total': '4324.60'}, {'lines': [['256.131', 4, '1'], ['26.633', 1, '0.5'], ['222.075', 4, '0'], ['1.858', 5, '1'], ['68.587', 10, '0.1'], ['84.715', 11, '0.25'], ['289.239', 11, '0.25'], ['294.076', 0, '0']], 'tax_rate': '0', 'line_totals': ['0.00', '13.32', '888.30', '0.00', '617.28', '698.90', '2386.22', '0.00'], 'total': '4604.02'}, {'lines': [['286.046', 0, '1'], ['193.575', 9, '0.33']], 'tax_rate': '0.005', 'line_totals': ['0.00', '1167.26'], 'total': '1173.10'}, {'lines': [['66.818', 0, '0.1'], ['191.182', 12, '1'], ['20.916', 5, '0.05'], ['357.599', 3, '0.5'], ['53.893', 5, '1'], ['293.54', 6, '0.33']], 'tax_rate': '0.5', 'line_totals': ['0.00', '0.00', '99.35', '536.40', '0.00', '1180.03'], 'total': '2723.67'}, {'lines': [['485.366', 3, '1'], ['85.197', 12, '1']], 'tax_rate': '0.005', 'line_totals': ['0.00', '0.00'], 'total': '0.00'}, {'lines': [['12.995', 2, '0.5'], ['484.697', 5, '1'], ['488.015', 6, '1'], ['351.224', 11, '1'], ['130.111', 4, '0.05'], ['412.811', 11, '0']], 'tax_rate': '0.125', 'line_totals': ['13.00', '0.00', '0.00', '0.00', '494.42', '4540.92'], 'total': '5679.38'}, {'lines': [], 'tax_rate': '1', 'line_totals': [], 'total': '0.00'}]

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
