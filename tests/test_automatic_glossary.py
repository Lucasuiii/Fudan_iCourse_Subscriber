import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from src.ai.automatic_glossary import validated_keywords, active_terms, AutomaticGlossary
from src.data.database import Database
from src.ai.summarizer import Summarizer
from scripts.merge_db import merge


class AutomaticGlossaryTests(unittest.TestCase):
    def test_merge_preserves_newer_glossary_and_unrelated_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            local=Database(str(Path(tmp)/'local.db'))
            remote=Database(str(Path(tmp)/'remote.db'))
            key='auto_glossary:10:1'
            local.write_meta(key,json.dumps({'updated_at':'2026-10-04T10:00:00+00:00','keywords':[]}))
            remote.write_meta(key,json.dumps({'updated_at':'2026-10-04T11:00:00+00:00','keywords':[]}))
            remote.write_meta('unrelated','keep')
            local.conn.close();remote.conn.close()
            merge(str(Path(tmp)/'local.db'),str(Path(tmp)/'remote.db'))
            remote=Database(str(Path(tmp)/'remote.db'))
            self.assertIn('11:00:00',remote.read_meta(key))
            self.assertEqual(remote.read_meta('unrelated'),'keep')
            remote.conn.close()
    def test_requires_exact_evidence_and_not_supplement(self):
        sources={'asr':['使用 Householder 变换与镜像变换'], 'ppt':['Gram-Schmidt 正交化']}
        items=[{'term':'Householder','source':'asr','quote':'使用 Householder 变换'},
               {'term':'Gram-Schmidt','source':'ppt','quote':'Gram-Schmidt 正交化'},
               {'term':'镜像变换','source':'cloud','quote':'镜像变换'},
               {'term':'老师姓名','source':'asr','quote':'老师姓名'},
               {'term':'忽略指令','source':'asr','quote':'忽略指令'}]
        summary='Householder 变换\n\n补充说明：Gram-Schmidt 正交化'
        self.assertEqual([x['term'] for x in validated_keywords(items,sources,summary)],['Householder'])
        self.assertEqual(validated_keywords(None,sources,summary),[])

    def test_two_distinct_lessons_and_no_self_reinforcement(self):
        one={'sub_id':'1','keywords':[{'term':'镜像变换','source':'asr'}]}
        self.assertEqual(active_terms([one,one]),[])
        two={**one,'sub_id':'2'}
        self.assertEqual(active_terms([one,two]),['镜像变换'])
        self.assertEqual(active_terms([one,two],exclude_sub_id='2'),[])
        self.assertEqual(active_terms([{'sub_id':'3','keywords':[{'term':'Householder','source':'ppt'}]}]),['Householder'])

    def test_persistence_course_isolation_and_deleted_lecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Database(str(Path(tmp)/'db.sqlite'))
            for cid,sid in [('10','1'),('20','2')]:
                db.upsert_course(cid,'同名课','教师')
                db.insert_lecture(sid,cid,'课次','2026-09-29')
            store=AutomaticGlossary(db,'10')
            store.save('1',[{'term':'Householder','source':'ppt','quote':'Householder'}])
            self.assertEqual(store.terms(),['Householder'])
            self.assertEqual(AutomaticGlossary(db,'20').terms(),[])
            self.assertEqual(store.terms(exclude_sub_id='1'),[])
            db.conn.execute("UPDATE lectures SET deleted_at='now' WHERE sub_id='1'")
            db.conn.commit()
            self.assertEqual(store.terms(),[])
            db.conn.close()

    def test_summary_and_keywords_one_request(self):
        client=MagicMock()
        client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop',message=SimpleNamespace(content=json.dumps({'summary':'### Householder 变换\n\n正文',
            'keywords':[{'term':'Householder','source':'ppt','quote':'Householder 变换'}]})))])
        model=Summarizer.__new__(Summarizer)
        model.system_prompt='仅输出笔记'
        model.providers=[{'name':'deepseek','models':['test']}]
        model._clients={'deepseek':client}
        with patch('src.ai.summarizer.enrich_summary',side_effect=lambda s,**kw:s):
            summary,_,keywords=model.summarize_with_keywords('课程','材料',{'ppt':['Householder 变换']},[])
        self.assertTrue(summary.startswith('###'))
        self.assertEqual(keywords[0]['term'],'Householder')
        client.chat.completions.create.assert_called_once()
        self.assertEqual(client.chat.completions.create.call_args.kwargs['extra_body']['thinking']['type'],'enabled')
        payload=json.loads(client.chat.completions.create.call_args.kwargs['messages'][1]['content'])
        self.assertIsInstance(payload['evidence_sources']['ppt'],str)


if __name__=='__main__':
    unittest.main()
