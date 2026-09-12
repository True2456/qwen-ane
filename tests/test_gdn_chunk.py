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

    def test_correlated_keys_fp16(self):
        # Large alternating powers in a Neumann inverse lose precision here.
        rng = np.random.default_rng(29)
        q = np.full((1,32,16), .25, np.float16)
        k = q.copy()
        v = rng.normal(size=q.shape).astype(np.float16)
        g = np.ones((1,32), np.float16)
        state = rng.normal(size=(1,16,16)).astype(np.float16)
        for strength in [.25,.5,1.]:
            beta = np.full(g.shape,strength,np.float16)
            y,final = chunk(q,k,v,g,beta,state)
            ry,rs = chunk(*[a.astype(np.float64) for a in (q,k,v,g,beta,state)])
            self.assertTrue(np.isfinite(y).all() and np.isfinite(final).all())
            self.assertLess(np.linalg.norm(y-ry)/np.linalg.norm(ry),.005)
            self.assertLess(np.linalg.norm(final-rs)/np.linalg.norm(rs),.005)


if __name__=='__main__': unittest.main()
