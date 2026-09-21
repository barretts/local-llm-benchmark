"""Maintenance must not authorize itself from missing handles or partial proof."""
import copy
from types import SimpleNamespace
import unittest
from scripts import speculation_setup_contract as safety

class SetupContractTests(unittest.TestCase):
    def setUp(self):
        self.run=(*safety.RUN,1)
        self.proof=dict(run_id=safety.RUN[0],original_started=safety.RUN[1],original_deadline=safety.RUN[2],
            reason='user_authorized_speculation_integration',checkpoints={x:{'complete':True} for x in safety.IDS})
        self.pins=copy.deepcopy(safety.PROCESS_PINS)
        self.closed=[]
    def handle(self,pid):
        created=next(x['created_filetime'] for x in self.pins if x['pid']==pid)
        return SimpleNamespace(pid=pid,created=created,close=lambda:self.closed.append(pid))
    def test_only_exact_finished_authorized_checkpoint_is_ready(self):
        self.assertTrue(safety.checkpoint_ready(self.run,self.proof,0))
        self.assertFalse(safety.checkpoint_ready((*safety.RUN,0),self.proof,0))
        self.assertFalse(safety.checkpoint_ready(self.run,self.proof,1))
        self.assertFalse(safety.checkpoint_ready((self.run[0],self.run[1],self.run[2]+1,1),self.proof,0))
        for mutate in (lambda p:p.update(checkpoints={}),lambda p:p['checkpoints'].pop(safety.IDS[0]),
            lambda p:p['checkpoints'][safety.IDS[0]].update(complete=False),lambda p:p.update(reason='other_stop')):
            changed=copy.deepcopy(self.proof);mutate(changed)
            self.assertFalse(safety.checkpoint_ready(self.run,changed,0))
    def test_binds_original_kernel_objects_without_process_mutation(self):
        handles=safety.original_handles(self.handle,self.pins,lambda:False,lambda pid:False)
        self.assertEqual([x.pid for x in handles],[26500,29624,28852]);self.assertEqual(self.closed,[])
    def test_reused_pid_is_left_untouched_only_after_authorized_pause(self):
        def replacement(pid):
            value=self.handle(pid)
            if pid==26500:value.created+=1000
            return value
        handles=safety.original_handles(replacement,self.pins,lambda:True,lambda pid:False)
        self.assertEqual([x.pid for x in handles],[29624,28852]);self.assertEqual(self.closed,[26500])
        with self.assertRaisesRegex(RuntimeError,'creation_changed'):
            safety.original_handles(replacement,self.pins,lambda:False,lambda pid:False)
    def test_failed_query_requires_both_finished_pause_and_actual_absence(self):
        def missing(pid):raise RuntimeError('query_failed')
        self.assertEqual(safety.original_handles(missing,self.pins,lambda:True,lambda pid:True),[])
        for paused,absent in ((False,True),(True,False)):
            with self.subTest(paused=paused,absent=absent),self.assertRaisesRegex(RuntimeError,'exit_unverifiable'):
                safety.original_handles(missing,self.pins,lambda:paused,lambda pid:absent)
    def test_unpinned_processes_cannot_become_maintenance_watchers(self):
        changed=copy.deepcopy(self.pins);changed[0]['pid']=99999
        with self.assertRaisesRegex(RuntimeError,'creation_pins_required'):
            safety.original_handles(self.handle,changed,lambda:True,lambda pid:True)

if __name__=='__main__':unittest.main()
