"""Data-free archive and reproduction checks. No Torch or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from single_object_artifacts import verified_members, restore
from single_object_summary import summarize
from bind_single_object_ft import bind


class SingleObjectReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.members=verified_members()

    def test_all_ft_parents_present_and_compatible(self):
        self.assertEqual(len(self.manifest['ft_dependencies']),114)
        paths={r['weights_path'] for r in self.manifest['ft_dependencies']}
        self.assertEqual(len(paths),78)
        for dep in self.manifest['ft_dependencies']:
            target=json.loads((ROOT/dep['config']).read_text())
            parent_path=str(Path(dep['weights_path']).parent/'config.json').replace('\\','/')
            parent=json.loads(self.members[parent_path])
            for key in ['seed','features_to_use','model']:
                self.assertEqual(target.get(key),parent.get(key),(dep['config'],key))
            if not target.get('model'):
                self.assertEqual(target.get('hidden_layers',[32,16]),parent.get('hidden_layers',[32,16]))

    def test_main_benchmark_values(self):
        rows={r['key']:r for r in summarize()}
        expected={'e2v4f_or':(67.97,.55),'e1v4f_ft':(69.61,.29),
                  'e3v2f_ft':(69.08,.35),'mlp_ctf_if':(68.40,.21),'v3f_or':(67.47,.22)}
        for key,(mean,se) in expected.items():
            self.assertEqual(round(rows[key]['global_mean'],2),mean)
            self.assertEqual(round(rows[key]['global_SE'],2),se)

    def test_restore_is_idempotent_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            data={'experiments/a/file':b'original'}
            self.assertEqual(restore(data,temp,'experiments/'),1)
            self.assertEqual(restore(data,temp,'experiments/'),0)
            with self.assertRaises(FileExistsError):
                restore({'experiments/a/file':b'changed'},temp,'experiments/')
            self.assertEqual((Path(temp)/'experiments/a/file').read_bytes(),b'original')

    def test_restore_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                restore({'experiments/../../escape':b'x'},temp,'experiments/')

    def test_binding_rejects_wrong_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); parent=root/'parent'; parent.mkdir()
            cfg={'seed':42,'features_to_use':['core_tensors'],'hidden_layers':[32,16],
                 'loss':{'name':'constrained_trigger'}}
            (root/'ft.json').write_text(json.dumps(cfg))
            (parent/'config.json').write_text(json.dumps({**cfg,'seed':123}))
            (parent/'model_weights.pt').write_bytes(b'not loaded by this test')
            with self.assertRaisesRegex(ValueError,'seed'):
                bind(root/'ft.json',parent,root/'out.json')

    def test_binding_writes_only_new_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); parent=root/'parent'; parent.mkdir()
            cfg={'seed':42,'features_to_use':['core_tensors'],'hidden_layers':[32,16],
                 'loss':{'name':'constrained_trigger'}}
            (root/'ft.json').write_text(json.dumps(cfg))
            (parent/'config.json').write_text(json.dumps({**cfg,'loss':{'name':'bce'}}))
            (parent/'model_weights.pt').write_bytes(b'not loaded by this test')
            bind(root/'ft.json',parent,root/'out.json')
            self.assertEqual(json.loads((root/'ft.json').read_text()),cfg)
            out=json.loads((root/'out.json').read_text())
            self.assertEqual(out['initialization']['weights_path'],(parent/'model_weights.pt').as_posix())
            with self.assertRaises(FileExistsError): bind(root/'ft.json',parent,root/'out.json')


if __name__=='__main__': unittest.main()
