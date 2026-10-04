import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from scripts.qwen_quality import context_echo,select_rescue_windows,review_quality,usable_ppt


class QualityTests(unittest.TestCase):
    def test_echo_is_not_single_term(self):
        context='数值算法。术语：良态问题、病态问题、扰动、Hilbert矩阵、逆矩阵、条件数、delta、范数。'
        self.assertTrue(context_echo(context,context))
        self.assertTrue(context_echo(context.split('术语：')[1],context))
        self.assertFalse(context_echo('希尔伯特矩阵是病态矩阵。',context))
        self.assertFalse(context_echo('',context))

    def test_cloud_budget_and_actual_windows(self):
        chunks=[{'start':0,'end':122},{'start':120,'end':242},{'start':240,'end':360}]
        windows=[(i,i+20) for i in range(0,360,25)]
        result=select_rescue_windows(chunks,[0,1,2],windows)
        self.assertLessEqual(sum(x['end_ms']-x['start_ms'] for x in result),120000)
        self.assertLessEqual(len(result),10)
        self.assertTrue(all(x['end_ms']-x['start_ms']<=60000 for x in result))
        self.assertEqual({x['chunk_id'] for x in result},{0,1,2})
        self.assertEqual(len(result),3)
        for x in result:
            chunk=chunks[x['chunk_id']]
            self.assertTrue(chunk['start']*1000<=x['start_ms']<x['end_ms']<=chunk['end']*1000)
            self.assertTrue(any(a*1000<x['end_ms'] and x['start_ms']<b*1000 for a,b in windows))
        self.assertEqual(select_rescue_windows(chunks,[0],[]),[])

    def test_long_silence_not_bridged(self):
        r=select_rescue_windows([{'start':0,'end':120}],[0],[(0,10),(90,100)])
        self.assertTrue(all(x['end_ms']-x['start_ms']<=10000 for x in r))

    def test_desktop_and_stale_page_filtered(self):
        self.assertFalse(usable_ppt('此电脑 回收站 巡检.exe 多媒体值班室',0))
        self.assertFalse(usable_ppt('矩阵逆与条件数',-5220))
        self.assertTrue(usable_ppt('矩阵逆与条件数',-100))

    def test_review_rejects_invented_ids_and_preserves_text(self):
        client=MagicMock()
        client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({'suspects':[
                {'id':True,'reason':'bad'},{'id':99,'reason':'bad'},
                {'id':0,'reason':'术语不通顺'},{'id':0,'reason':'duplicate'}]})))])
        r={'full_chunks':[{'start':0,'end':120,'text':'原字幕'}]}
        self.assertEqual(review_quality(client,'model',r,{}),[{'id':0,'reason':'术语不通顺'}])
        self.assertEqual(r['full_chunks'][0]['text'],'原字幕')
        client.chat.completions.create.assert_called_once()


if __name__=='__main__':
    unittest.main()
