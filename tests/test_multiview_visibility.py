"""Regression checks for multi-view visibility, occlusion, and frame counting."""
import sys,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from processing.rgb_recovery.fusion import project_visibility

class VisibilityTests(unittest.TestCase):
    K=np.array([[10.,0,5],[0,10,5],[0,0,1]])
    xyz=np.array([[0.,0,.5]])

    def test_batched_votes_and_colours_equal_single_batch(self):
        points=np.array([[0,0,.5],[0,0,.7],[0,0,.3],[.01,0,.5],[0,.01,.5]])
        frames=[self.frame(1),self.frame(2,.7),self.frame(3,.3)]
        full=project_visibility(points,frames,self.K,chunk_size=100)
        chunks=project_visibility(points,frames,self.K,chunk_size=2)
        for expected,actual in zip(full,chunks):np.testing.assert_array_equal(actual,expected)

    def test_shared_plane_uses_central_views_in_metric_coordinates(self):
        from processing.rgb_recovery.fusion import foreground_plane
        frames=[]
        for i in range(11):
            transform=np.eye(4);transform[0,3]=(i-5)*.01;transform[2,3]=.02
            frames.append(dict(frame=i,T=transform))
        cfg=dict(foreground='auto',reference=5,dist=[0]*5,width=11,height=11)
        def read(root,frame,K,maps):
            return dict(foreground_limit=.8 if 2<=frame<=8 else 1.4)
        with patch('processing.rgb_recovery.fusion.read_frame',side_effect=read):
            plane=foreground_plane(cfg,frames,'.',self.K)
        self.assertAlmostEqual(plane['maximum'],.82)
        self.assertEqual(set(plane['source_frames']),set(range(2,9)))
        self.assertEqual(plane['normal'],[0,0,1])

    def frame(self,index,depth=.5,mask=True):
        return dict(frame=index,T=np.eye(4),z=np.full((11,11),depth),votes=np.full((11,11),4),mask=np.full((11,11),mask),rgb=np.tile(np.array([10,20,30],dtype=np.uint8),(11,11,1)))

    def test_distinct_views_support_measured_surface(self):
        s,c,rgb=project_visibility(self.xyz,[self.frame(i) for i in range(3)],self.K)
        self.assertEqual(s.tolist(),[3]);self.assertEqual(c.tolist(),[0]);self.assertEqual(rgb.tolist(),[[30,20,10]])

    def test_missing_depth_is_not_support_or_contradiction(self):
        s,c,_=project_visibility(self.xyz,[self.frame(1,np.nan)],self.K)
        self.assertEqual(s.tolist(),[0]);self.assertEqual(c.tolist(),[0])

    def test_occluded_surface_is_not_a_free_space_conflict(self):
        s,c,_=project_visibility(self.xyz,[self.frame(1,.4)],self.K)
        self.assertEqual(s.tolist(),[0]);self.assertEqual(c.tolist(),[0])

    def test_empty_space_contradicts_a_phantom_foreground_surface(self):
        s,c,_=project_visibility(self.xyz,[self.frame(1,.7)],self.K)
        self.assertEqual(s.tolist(),[0]);self.assertEqual(c.tolist(),[1])

    def test_nonplant_scene_surface_cannot_support_plant(self):
        s,c,_=project_visibility(self.xyz,[self.frame(1,.5,False)],self.K)
        self.assertEqual(s.tolist(),[0]);self.assertEqual(c.tolist(),[0])

    def test_same_frame_cannot_vote_twice(self):
        with self.assertRaises(ValueError):
            project_visibility(self.xyz,[self.frame(1),self.frame(1)],self.K)

    def test_missing_center_depth_does_not_make_supported_edge_black(self):
        f=self.frame(1);f['z'][5,5]=np.nan;f['votes'][5,5]=0
        s,c,rgb=project_visibility(self.xyz,[f],self.K)
        self.assertEqual(s.tolist(),[1]);self.assertEqual(rgb.tolist(),[[30,20,10]])

if __name__=='__main__':unittest.main()
