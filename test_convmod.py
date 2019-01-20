from unittest import TestCase
import unittest
import pandas as pd
import torch
from .Imodules import IConvMod
from .logmem import log_mem

class TestConvMod(TestCase):

    @classmethod
    def setUpClass(cls):
        """
        f_f -> conv bn act (conv bn act conv) bn act  +skip -skip --> 10 base - 1 base
        f_t -> conv (conv/2 + conv/2) *2 - 2 conv +skip --> 2 base + 4 base/2 - 2base
        t_f -> conv bn act  (conv/2 + bn/2 + act/2 + conv/2) *2  + bn + act  - 1 conv ---> 5 base + 8 base/2 - 1 base
        t_t -> conv (conv/4 + conv/4) * 8 + 2 cp  - 2 cp - 2 conv/2 - 1 conv ---> 1 base + 2 base/2 + 8 base/4 - 1 base - 4 base/2
        """
        mem_log = []
        cls.base = 20
        for invert in [False, True]:
            for skip_invert in [False, True]:
                exp = f"ConvMod_skip_{skip_invert}_invert_{invert}"
                inp = torch.ones(1,32,40,64,64).cuda(0)
                mod = IConvMod(32, 32,invert=invert, skip_invert=skip_invert).cuda(0)
                log_mem(mod, inp, mem_log, exp)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        df = pd.DataFrame(mem_log)
        for exp in df.exp.drop_duplicates():
            df.loc[df.exp==exp, 'mem_diff'] = ((df[df.exp==exp].sort_values(by="call_idx").mem_all.diff())//2**20).fillna(0).map(int).values
            df.loc[df.exp==exp, 'cum_mem'] = df[df.exp==exp].sort_values(by="call_idx").mem_diff.cumsum()
            df.loc[df.exp==exp, 'is_fwd'] = df[df.exp==exp].transpose().apply(lambda x: x.loc['call_idx'] < df[(df.exp==exp) & (df.hook_type=="bwd")].call_idx.min()).values
        cls.df = df

    def test_no_inv(self):
        exp_t_t = "ConvMod_skip_False_invert_False"
        df_exp = self.df.loc[self.df.exp==exp_t_t]

        df_exp_fwd = df_exp[df_exp.is_fwd]

        self.assertEqual( df_exp_fwd[df_exp.mem_diff == self.base].call_idx.count(), 10)
        self.assertEqual( df_exp_fwd[df_exp.mem_diff == self.base/2].call_idx.count(), 0)
        self.assertEqual( df_exp_fwd[df_exp.mem_diff == self.base/4].call_idx.count(), 0)

        self.assertEqual( df_exp_fwd[df_exp.mem_diff == -self.base/4].call_idx.count(), 0)
        self.assertEqual( df_exp_fwd[df_exp.mem_diff == -self.base/2].call_idx.count(), 0)
        self.assertEqual( df_exp_fwd[df_exp.mem_diff == -self.base].call_idx.count(), 1)

    def test_skip_inv(self):

        exp_t_f = "ConvMod_skip_True_invert_False"
        df_exp = self.df.loc[self.df.exp==exp_t_f]

        df_exp_fwd = df_exp[df_exp.is_fwd]

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base].call_idx.count(), 5)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/2].call_idx.count(), 8)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/4].call_idx.count(), 0)

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/4].call_idx.count(), 0)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/2].call_idx.count(), 0)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base].call_idx.count(), 1)

    def test_inv_conv(self):

        exp_f_t = "ConvMod_skip_False_invert_True"
        df_exp = self.df.loc[self.df.exp==exp_f_t]

        df_exp_fwd = df_exp[df_exp.is_fwd]

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base].call_idx.count(), 2)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/2].call_idx.count(), 4)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/4].call_idx.count(), 0)

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/4].call_idx.count(), 0)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/2].call_idx.count(), 0)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base].call_idx.count(), 2)


    def test_full_inv(self):

        exp_t_t = "ConvMod_skip_True_invert_True"
        df_exp = self.df.loc[self.df.exp==exp_t_t]

        df_exp_fwd = df_exp[df_exp.is_fwd]

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base].call_idx.count(), 1)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/2].call_idx.count(), 2)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == self.base/4].call_idx.count(), 8)

        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/4].call_idx.count(), 0)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base/2].call_idx.count(), 4)
        self.assertEqual(df_exp_fwd[df_exp.mem_diff == -self.base].call_idx.count(), 1)

if __name__ == '__main__':
    unittest.main()