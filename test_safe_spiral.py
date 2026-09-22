import os
import unittest
import torch
import torch.nn.functional as F
from safe_spiral import bounded_spiral_align

def order(coords, origin):
    oy, ox = origin
    def key(j):
        y, x = coords[j]
        dy, dx = y-oy, x-ox
        r = max(abs(dy), abs(dx))
        if r == 0: return (0, 0, j)
        if dx == r and dy > -r: off = dy+r-1
        elif dy == r: off = 2*r+(r-1-dx)
        elif dx == -r: off = 4*r+(r-1-dy)
        else: off = 6*r+(dx+r-1)
        return r, off, j
    return sorted(range(len(coords)), key=key)

class Model:
    spatial_gamma = .25
    spatial_sigma = 8.
    patch_stride = 250.
    align_topk = 64
    temperature = 1.
    mode = 'both'
    def _similarity(self, he, ihc, hm, im):
        f = F.normalize(ihc, dim=1) @ F.normalize(he, dim=1).T
        x, y = im.flatten(1), hm.flatten(1)
        xm, ym = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
        xc, yc = x-xm, y-ym
        d = max(x.shape[1]-1, 1)
        cov = xc @ yc.T / d
        xv, yv = xc.square().sum(1,keepdim=True)/d, yc.square().sum(1,keepdim=True).T/d
        h = ((2*xm@ym.T+.01**2)/(xm.square()+ym.T.square()+.01**2)
             * (2*cov+.03**2)/(xv+yv+.03**2))
        b = f if self.mode == 'feature' else h if self.mode == 'morph' else .5*(f+h)
        return f, h, b, torch.tensor([.5,.5],device=he.device)

class Tests(unittest.TestCase):
    def compare(self, m, coords, anchors, same=False, device='cpu'):
        torch.manual_seed(42)
        he = torch.rand(len(coords), 16, device=device, requires_grad=True)
        ihc = torch.rand(len(anchors),16,device=device)
        hm = torch.rand(len(coords),8,8,device=device)
        im = torch.rand(len(anchors),8,8,device=device)
        if same:
            he = torch.ones_like(he).requires_grad_()
            ihc = torch.ones_like(ihc)
            hm, im = torch.ones_like(hm), torch.ones_like(im)
        args = (m,he,ihc,hm,im,coords,anchors,order)
        os.environ['VGMS_SAFE_STOP']='0'
        full=bounded_spiral_align(*args)
        full[0].sum().backward()
        grad=he.grad.clone()
        he.grad=None
        os.environ['VGMS_SAFE_STOP']='1'
        safe=bounded_spiral_align(*args)
        safe[0].sum().backward()
        self.assertTrue(torch.equal(full[1],safe[1]))
        torch.testing.assert_close(full[2],safe[2],rtol=0,atol=0)
        torch.testing.assert_close(full[0],safe[0],rtol=0,atol=0)
        torch.testing.assert_close(grad,he.grad,rtol=0,atol=0)
        return safe
    def test_pruning_modes_and_batches(self):
        coords=[(y*250,x*250) for y in range(-5,6) for x in range(-5,6)]
        coords += [(25000+j*250, 30000) for j in range(250)]
        for mode in ['both','feature','morph']:
            for batch in [1,17,128]:
                os.environ['VGMS_SEARCH_BATCH']=str(batch)
                m=Model(); m.mode=mode
                self.compare(m,coords,[(0,0),(250,500)])
                self.assertLess(m.last_search_stats['scored_fraction'],1.)
    def test_no_penalty_ties_and_duplicates(self):
        os.environ['VGMS_SEARCH_BATCH']='17'
        m=Model(); m.spatial_gamma=0
        out=self.compare(m,[(i%30*250,i//30*250) for i in range(200)],[(0,0)],same=True)
        self.assertEqual(out[1][0].tolist(),list(range(64)))
        self.assertEqual(m.last_search_stats['scored_fraction'],1.)
        m=Model(); self.compare(m,[(0,0)]*180,[(0,0)],same=True)
    def test_small_k_and_nonmonotone_euclidean_distance(self):
        os.environ['VGMS_SEARCH_BATCH']='3'
        for size in [1,7,63,64,65]:
            m=Model()
            self.compare(m,[(i*250,-i*250) for i in range(size)],[(300,-100)])
        m=Model(); m.align_topk=1
        self.compare(m,[(2500,2500),(2750,0),(10000,0),(0,0)],[(0,0)])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA server required')
    def test_cuda_amp(self):
        os.environ['VGMS_SEARCH_BATCH']='128'
        with torch.autocast('cuda',dtype=torch.float16):
            self.compare(Model(),[(i*250,0) for i in range(400)],[(0,0)],device='cuda')

if __name__=='__main__': unittest.main(verbosity=2)
