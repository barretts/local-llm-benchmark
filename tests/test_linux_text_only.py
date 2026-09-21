"""Text-only image controls are real flags, never assumed accepted."""
import unittest
from tests import test_container_adapter as fixtures


class LinuxTextOnlyTests(unittest.TestCase):
    setUp=fixtures.ContainerAdapterTests.setUp
    tearDown=fixtures.ContainerAdapterTests.tearDown
    adapter=fixtures.ContainerAdapterTests.adapter

    def test_both_linux_engines_require_actual_help_for_encoder_loading_control(self):
        for engine in ('vllm','sglang'):
            with self.subTest(engine=engine):
                adapter=self.adapter(engine);adapter.help=fixtures.FLAGS
                self.assertIs(adapter.requested['language_model_only'],True)
                self.assertIn('--language-model-only',adapter._engine_args())
                adapter.help=fixtures.FLAGS.replace('--language-model-only','')
                with self.assertRaisesRegex(RuntimeError,'unsupported_exact_image_flag:--language-model-only'):
                    adapter._engine_args()

    def test_disabled_or_invalid_control_is_explicit(self):
        adapter=self.adapter('sglang',settings={'language_model_only':False});adapter.help=fixtures.FLAGS
        self.assertNotIn('--language-model-only',adapter._engine_args())
        with self.assertRaisesRegex(ValueError,'invalid_language_model_only_control'):
            self.adapter('vllm',settings={'language_model_only':'true'})


if __name__=='__main__':unittest.main()
