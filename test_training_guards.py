"""Exercise final inference guard from the trainer without loading models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

SOURCE = Path(__file__).resolve().parent/'phase3/train_cogvideox_lora.py'


class TrainingGuards(unittest.TestCase):
    def test_final_validation_guard(self):
        tree=ast.parse(SOURCE.read_text())
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.If)
                and ast.unparse(n.test)=='args.validation_prompt and args.num_validation_videos > 0']
        self.assertEqual(len(guards),1)
        code=compile(ast.Module(body=[guards[0]],type_ignores=[]),str(SOURCE),'exec')
        for prompt,count,expected in [(None,0,0),('wheel',0,0),(None,1,0),('wheel',1,1)]:
            pipeline=MagicMock();validate=MagicMock(return_value=['video'])
            args=SimpleNamespace(validation_prompt=prompt,num_validation_videos=count,
                pretrained_model_name_or_path='fixture',revision='fixed',variant=None,
                enable_slicing=False,enable_tiling=False,output_dir='fixture',
                lora_alpha=128,rank=64,validation_prompt_separator='|||',
                guidance_scale=6,use_dynamic_cfg=False,height=480,width=720)
            scope=dict(args=args,CogVideoXPipeline=pipeline,CogVideoXDPMScheduler=MagicMock(),
                weight_dtype=None,log_validation=validate,accelerator=None,epoch=0,validation_outputs=[])
            exec(code,scope)
            self.assertEqual(pipeline.from_pretrained.call_count,expected)
            self.assertEqual(validate.call_count,expected)

    def test_sampler_has_explicit_generator(self):
        tree=ast.parse(SOURCE.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DataLoader']
        self.assertEqual(len(calls),1)
        self.assertTrue(any(k.arg=='generator' for k in calls[0].keywords))


if __name__=='__main__':unittest.main()
