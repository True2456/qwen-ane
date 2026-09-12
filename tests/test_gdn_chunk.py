"""Exercise nonzero incoming state, updated-state output, and gate edge cases."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
from gdn_chunk_reference import chunk


class ChunkTests(unittest.TestCase):
    def test_published_form_matches_recurrence(self):
        rng = np.random.default_rng(103)
        for c in [1,4,8,16,32]:
            for gate_case in ['random','zero','one','tiny']:
                q,k,v = [rng.normal(size=(3,c,16)) for _ in range(3)]
                k /= np.linalg.norm(k, axis=-1, keepdims=True)
                q /= np.linalg.norm(q, axis=-1, keepdims=True)
                g = rng.uniform(.01,1,(3,c))
                if gate_case=='zero': g[:,c//2]=0
                if gate_case=='one': g[:]=1
                if gate_case=='tiny': g[:]=1e-8
                beta = rng.uniform(0,1,(3,c))
                state = rng.normal(size=(3,16,16))
                y,final=chunk(q,k,v,g,beta,state)
                ref=state.copy(); ys=[]
                for t in range(c):
                    ref *= g[:,t,None,None]
                    mem = np.einsum('hvk,hk->hv',ref,k[:,t])
                    delta=(v[:,t]-mem)*beta[:,t,None]
                    ref += delta[:,:,None]*k[:,t,None,:]
                    ys.append(np.einsum('hvk,hk->hv',ref,q[:,t]))
                np.testing.assert_allclose(y,np.stack(ys,1),atol=2e-14,rtol=2e-12)
                np.testing.assert_allclose(final,ref,atol=2e-14,rtol=2e-12)

if __name__=='__main__': unittest.main()
