import unittest
from scripts.qwen_quality import aligned_quote_span,aligned_rescue_intervals


class AudioAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.text='开始这里讨论舍入误差如何分析结束'
        self.chunk={'start':100,'end':130,'text':self.text}
        self.quote='这里讨论舍入误差如何分析'
        self.items=[{'text':x,'start':i,'end':i+0.5} for i,x in enumerate(self.text)]

    def test_quoted_span_uses_audio_times_not_midpoint(self):
        a,b=aligned_quote_span(self.chunk,self.quote,self.items,[(100,130)])
        self.assertEqual(a,102)
        self.assertEqual(b,113.5)

    def test_bad_alignment_is_rejected(self):
        for items in (self.items[:-1],
                      [{**x,'start':0,'end':0} for x in self.items],
                      [{**x,'start':float('nan')} for x in self.items],
                      list(reversed(self.items))):
            with self.assertRaises(ValueError):
                aligned_quote_span(self.chunk,self.quote,items,[(100,130)])
        with self.assertRaises(ValueError):
            aligned_quote_span(self.chunk,self.quote,self.items,[])

    def test_repeated_quote_never_guesses_position(self):
        with self.assertRaises(ValueError):
            aligned_quote_span({**self.chunk,'text':self.text+self.text},self.quote,self.items,[(100,130)])

    def test_padding_budget_and_no_overlap(self):
        chunks=[self.chunk]
        located=[{'id':0,'quote':self.quote,'start':102,'end':114}]
        intervals,accepted,rejected=aligned_rescue_intervals(chunks,located,budget=20)
        self.assertEqual(intervals[0]['start_ms'],100000)
        self.assertEqual(intervals[0]['end_ms'],117000)
        self.assertEqual(len(accepted),1)
        self.assertEqual(rejected,[])
        self.assertEqual(aligned_rescue_intervals(chunks,located,budget=5)[0],[])


if __name__=='__main__':
    unittest.main()
