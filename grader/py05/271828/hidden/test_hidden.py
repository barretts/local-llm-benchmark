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
CASES = [{'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['476.59', 1, '0.25'], ['433.655', 0, '0.5']], 'tax_rate': '0.005', 'line_totals': ['0.01', '0.01', '357.44', '0.00'], 'total': '359.25'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['3.448', 2, '0'], ['286.713', 3, '0.1'], ['456.775', 10, '0.33']], 'tax_rate': '0.125', 'line_totals': ['0.01', '0.01', '6.90', '774.13', '3060.39'], 'total': '4321.62'}, {'lines': [['0.005', 1, '0'], ['0.005', 1, '0'], ['319.192', 5, '0.1'], ['47.039', 7, '1']], 'tax_rate': '0.2', 'line_totals': ['0.01', '0.01', '1436.36', '0.00'], 'total': '1723.66'}, {'lines': [['229.826', 10, '0.1'], ['175.881', 4, '0.05']], 'tax_rate': '0.005', 'line_totals': ['2068.43', '668.35'], 'total': '2750.46'}, {'lines': [], 'tax_rate': '0.125', 'line_totals': [], 'total': '0.00'}, {'lines': [['53.922', 4, '0.33'], ['304.446', 1, '0.05'], ['276.319', 9, '0.1'], ['490.451', 5, '0.33'], ['113.538', 0, '0'], ['485.492', 5, '0.05'], ['375.496', 11, '0.25'], ['228.152', 1, '0.33']], 'tax_rate': '0.5', 'line_totals': ['144.51', '289.22', '2238.18', '1643.01', '0.00', '2306.09', '3097.84', '152.86'], 'total': '14807.57'}, {'lines': [['369.488', 11, '0.5'], ['256.206', 5, '0.33'], ['409.578', 4, '0.5'], ['427.986', 4, '0'], ['315.313', 1, '0.1']], 'tax_rate': '0', 'line_totals': ['2032.18', '858.29', '819.16', '1711.94', '283.78'], 'total': '5705.35'}, {'lines': [['41.268', 12, '0.1']], 'tax_rate': '0.01', 'line_totals': ['445.69'], 'total': '450.15'}, {'lines': [['160.469', 8, '0.05'], ['194.667', 1, '0'], ['143.901', 9, '0.1'], ['252.384', 2, '0.25'], ['307.192', 10, '0']], 'tax_rate': '0.01', 'line_totals': ['1219.56', '194.67', '1165.60', '378.58', '3071.92'], 'total': '6090.63'}, {'lines': [['75.603', 2, '0.5'], ['357.387', 10, '0.33'], ['286.332', 11, '0'], ['62.41', 8, '0.05'], ['361.871', 2, '0.25'], ['257.064', 9, '0.05'], ['229.157', 9, '0.1'], ['345.973', 5, '0.25']], 'tax_rate': '0', 'line_totals': ['75.60', '2394.49', '3149.65', '474.32', '542.81', '2197.90', '1856.17', '1297.40'], 'total': '11988.34'}, {'lines': [['235.5', 12, '0'], ['422.267', 9, '0']], 'tax_rate': '0.2', 'line_totals': ['2826.00', '3800.40'], 'total': '7951.68'}, {'lines': [['102.637', 0, '0.33'], ['323.051', 4, '0'], ['248.576', 6, '0.1'], ['298.378', 1, '1']], 'tax_rate': '0', 'line_totals': ['0.00', '1292.20', '1342.31', '0.00'], 'total': '2634.51'}, {'lines': [['452.186', 12, '0.05'], ['180.255', 10, '0.33'], ['122.072', 10, '0.33'], ['135.803', 8, '0.33'], ['284.354', 8, '0.25'], ['214.897', 4, '0.33'], ['321.089', 10, '0.1'], ['472.542', 10, '0.25']], 'tax_rate': '1', 'line_totals': ['5154.92', '1207.71', '817.88', '727.90', '1706.12', '575.92', '2889.80', '3544.07'], 'total': '33248.64'}, {'lines': [['106.955', 0, '0.33'], ['71.924', 4, '1'], ['337.497', 12, '0.5'], ['410.843', 12, '1'], ['463.205', 12, '1'], ['494.394', 8, '0.1'], ['109.754', 1, '0.05']], 'tax_rate': '0', 'line_totals': ['0.00', '0.00', '2024.98', '0.00', '0.00', '3559.64', '104.27'], 'total': '5688.89'}, {'lines': [['237.592', 8, '0.5'], ['280.261', 8, '0.25'], ['187.579', 10, '0.05'], ['317.204', 11, '0'], ['185.353', 4, '0.33']], 'tax_rate': '0', 'line_totals': ['950.37', '1681.57', '1782.00', '3489.24', '496.75'], 'total': '8399.93'}, {'lines': [['468.294', 3, '0.25'], ['307.671', 12, '0.05'], ['183.045', 12, '1'], ['137.989', 12, '0.5'], ['8.67', 7, '0.1']], 'tax_rate': '0.005', 'line_totals': ['1053.66', '3507.45', '0.00', '827.93', '54.62'], 'total': '5470.88'}, {'lines': [['198.946', 0, '0.5']], 'tax_rate': '1', 'line_totals': ['0.00'], 'total': '0.00'}, {'lines': [['407.168', 2, '0.1'], ['358.261', 7, '1']], 'tax_rate': '0.01', 'line_totals': ['732.90', '0.00'], 'total': '740.23'}, {'lines': [['51.774', 5, '1'], ['292.342', 4, '0'], ['248.088', 12, '0.25'], ['30.5', 10, '0.05'], ['275.811', 11, '0.25']], 'tax_rate': '0.01', 'line_totals': ['0.00', '1169.37', '2232.79', '289.75', '2275.44'], 'total': '6027.02'}, {'lines': [['380.504', 0, '0.05'], ['210.423', 6, '1'], ['400.467', 6, '0.33'], ['478.098', 4, '0.5']], 'tax_rate': '0', 'line_totals': ['0.00', '0.00', '1609.88', '956.20'], 'total': '2566.08'}, {'lines': [['296.105', 10, '0.1'], ['431.119', 2, '0.25'], ['206.523', 3, '1'], ['60.895', 6, '0.33'], ['44.96', 7, '0.25']], 'tax_rate': '0.125', 'line_totals': ['2664.95', '646.68', '0.00', '244.80', '236.04'], 'total': '4266.53'}, {'lines': [['302.182', 10, '0'], ['9.473', 8, '0.33'], ['334.664', 11, '0.1'], ['376.453', 12, '0.25'], ['469.053', 0, '0.25']], 'tax_rate': '0.01', 'line_totals': ['3021.82', '50.78', '3313.17', '3388.08', '0.00'], 'total': '9871.59'}, {'lines': [], 'tax_rate': '0.01', 'line_totals': [], 'total': '0.00'}, {'lines': [['478.94', 10, '1'], ['91.303', 12, '1'], ['398.579', 4, '0.25'], ['160.398', 6, '0'], ['321.842', 11, '0.33'], ['28.248', 1, '0.1']], 'tax_rate': '1', 'line_totals': ['0.00', '0.00', '1195.74', '962.39', '2371.98', '25.42'], 'total': '9111.06'}, {'lines': [['341.54', 12, '0.05'], ['30.425', 11, '0.33'], ['193.186', 1, '0.05'], ['73.538', 2, '0.25']], 'tax_rate': '0.01', 'line_totals': ['3893.56', '224.23', '183.53', '110.31'], 'total': '4455.75'}, {'lines': [['329.931', 4, '0.25'], ['467.636', 3, '0.05'], ['4.213', 1, '0'], ['485.808', 12, '0.33'], ['20.365', 7, '0'], ['342.405', 8, '0.05'], ['247.504', 2, '0.25']], 'tax_rate': '0', 'line_totals': ['989.79', '1332.76', '4.21', '3905.90', '142.56', '2602.28', '371.26'], 'total': '9348.76'}, {'lines': [['245.936', 6, '0']], 'tax_rate': '1', 'line_totals': ['1475.62'], 'total': '2951.24'}, {'lines': [['248.01', 5, '0.5'], ['149.977', 5, '0.33'], ['309.285', 0, '0.25'], ['237.901', 3, '0.25']], 'tax_rate': '0.2', 'line_totals': ['620.03', '502.42', '0.00', '535.28'], 'total': '1989.28'}, {'lines': [['277.264', 10, '0.25'], ['434.068', 2, '0.33']], 'tax_rate': '0', 'line_totals': ['2079.48', '581.65'], 'total': '2661.13'}, {'lines': [['43.006', 6, '0'], ['288.984', 8, '0.05'], ['401.754', 3, '0.33'], ['334.908', 2, '0.5'], ['355.668', 2, '0.1']], 'tax_rate': '0.2', 'line_totals': ['258.04', '2196.28', '807.53', '334.91', '640.20'], 'total': '5084.35'}, {'lines': [['426.46', 9, '0.1'], ['492.244', 2, '0.5'], ['167.433', 10, '0.5'], ['254.244', 11, '0.25'], ['420.042', 0, '0'], ['353.496', 4, '0.05']], 'tax_rate': '1', 'line_totals': ['3454.33', '492.24', '837.17', '2097.51', '0.00', '1343.28'], 'total': '16449.06'}, {'lines': [['317.395', 4, '0']], 'tax_rate': '0.5', 'line_totals': ['1269.58'], 'total': '1904.37'}, {'lines': [['124.274', 12, '0.5'], ['270.74', 0, '0.1'], ['455.594', 3, '0.1'], ['151.695', 8, '0.5']], 'tax_rate': '0', 'line_totals': ['745.64', '0.00', '1230.10', '606.78'], 'total': '2582.52'}, {'lines': [['378.841', 6, '0.05']], 'tax_rate': '0', 'line_totals': ['2159.39'], 'total': '2159.39'}, {'lines': [['175.662', 9, '0.25'], ['52.522', 12, '0.33'], ['268.667', 5, '0'], ['216.481', 6, '0.25'], ['308.148', 10, '0.5'], ['306.716', 7, '0.5']], 'tax_rate': '0.01', 'line_totals': ['1185.72', '422.28', '1343.34', '974.16', '1540.74', '1073.51'], 'total': '6605.15'}, {'lines': [['378.667', 1, '0.33'], ['209.912', 12, '0.1'], ['112.219', 8, '0'], ['465.461', 5, '0.5']], 'tax_rate': '0.2', 'line_totals': ['253.71', '2267.05', '897.75', '1163.65'], 'total': '5498.59'}, {'lines': [['370.635', 12, '0'], ['76.222', 9, '0.1'], ['361.305', 9, '1'], ['454.242', 5, '0.25']], 'tax_rate': '0.5', 'line_totals': ['4447.62', '617.40', '0.00', '1703.41'], 'total': '10152.65'}, {'lines': [['217.758', 12, '1'], ['147.763', 1, '1'], ['364.188', 12, '0'], ['422.7', 9, '0'], ['296.692', 5, '0.25']], 'tax_rate': '0.005', 'line_totals': ['0.00', '0.00', '4370.26', '3804.30', '1112.60'], 'total': '9333.60'}, {'lines': [['248.295', 3, '0.33'], ['193.854', 4, '0.1'], ['297.899', 10, '0.25'], ['208.469', 9, '1'], ['255.92', 4, '0.5'], ['75.443', 2, '0']], 'tax_rate': '0', 'line_totals': ['499.07', '697.87', '2234.24', '0.00', '511.84', '150.89'], 'total': '4093.91'}, {'lines': [['359.97', 11, '0.5'], ['385.652', 4, '0.1'], ['181.974', 12, '0.33']], 'tax_rate': '0.5', 'line_totals': ['1979.84', '1388.35', '1463.07'], 'total': '7246.89'}, {'lines': [['294.721', 3, '0.33'], ['382.586', 0, '0.33'], ['440.189', 2, '0.05'], ['135.742', 7, '1'], ['167.427', 12, '0'], ['24.552', 10, '0']], 'tax_rate': '0.005', 'line_totals': ['592.39', '0.00', '836.36', '0.00', '2009.12', '245.52'], 'total': '3701.81'}, {'lines': [['466.99', 5, '0.1'], ['327.679', 8, '0.05']], 'tax_rate': '0.125', 'line_totals': ['2101.46', '2490.36'], 'total': '5165.80'}, {'lines': [['124.652', 1, '0.25'], ['409.458', 7, '0.33'], ['197.402', 5, '0.1'], ['454.918', 6, '0.1'], ['486.497', 10, '0.05'], ['202.512', 11, '0'], ['65.977', 3, '0.05']], 'tax_rate': '0.125', 'line_totals': ['93.49', '1920.36', '888.31', '2456.56', '4621.72', '2227.63', '188.03'], 'total': '13945.61'}, {'lines': [['23.906', 12, '0.1'], ['441.508', 5, '0.1'], ['448.807', 2, '0.5'], ['2.931', 1, '0.05'], ['478.031', 0, '0.25'], ['141.826', 0, '0.1'], ['120.052', 5, '0.1'], ['135.653', 0, '0.1']], 'tax_rate': '0.5', 'line_totals': ['258.18', '1986.79', '448.81', '2.78', '0.00', '0.00', '540.23', '0.00'], 'total': '4855.19'}, {'lines': [], 'tax_rate': '0.005', 'line_totals': [], 'total': '0.00'}]

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
