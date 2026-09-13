"""Copy preserved owner/template and adapt only saved-audit source namespaces."""
from pathlib import Path
import difflib,hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'independent_plant_pending_publication_saved_actual_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,data):
    with (BASE/name).open('xb') as f:f.write(data)
def main():
    copies={};changed={};diff=''
    for name in ('verify_completion.py','preserved_saved_audit_template.ps1.txt','check_template.ps1'):
        put(name,(OLD/name).read_bytes());copies[name]=sha(BASE/name)
    for name in ('prepare_launch.py','test_launch.py','record_helper_preparation.py'):
        text=(OLD/name).read_text()
        pairs=[('independent_plant_pending_publication_saved_audit_v1','independent_plant_pending_result_saved_audit_v1'),
            ('independent_plant_pending_publication_saved_actual_v1','independent_plant_pending_result_saved_actual_v1'),
            ('independent_pending_publication_saved_root_review_v1/review.json','independent_pending_result_saved_audit_review_v1/review.json'),
            ('81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0','009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48')]
        for old,new in pairs:text=text.replace(old,new)
        if name=='record_helper_preparation.py':
            text=text.replace("old=NEW/'independent_plant_clock_saved_actual_v1'", "old=NEW/'independent_plant_pending_publication_saved_actual_v1'")
            text=text.replace('len(source_map)==17','len(source_map)==20')
            text=text.replace("    assert review['required_clock_request_contract']=='pending_BUSY_same_job_max10_before_original_activation'", "    preparation=NEW/'independent_plant_pending_result_saved_audit_v1/source_preparation_v2.json'\n    assert Path(review['preparation_subject']['path']).resolve()==preparation.resolve()\n    assert review['preparation_subject']['sha256']==sha(preparation)=='09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'")
            text=text.replace('exact 17-file root review','exact 20-file independent review')
        if name=='test_launch.py':
            marker="    def test_literal_hash_required(self):"
            text=text.replace(marker,"    def test_mandatory_dispatch_argument_preserved(self):\n        text=render('a'*64)\n        self.assertIn('param([Parameter(Mandatory=$true)][string]$LaunchReceiptSha256)',text)\n\n"+marker)
        put(name,text.encode());changed[name]=sha(BASE/name)
        diff+=''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),text.splitlines(True),fromfile='prior_saved/'+name,tofile='pending_result_saved/'+name))
    put('helper_derivation.diff',diff.encode())
    put('helper_derivation.json',json.dumps(dict(unchanged=copies,changed=changed,original=OLD.as_posix(),source_only=True,actual_dispatch=False),indent=2).encode())
if __name__=='__main__':main()
