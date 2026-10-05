"""One bounded auth-only probe: no database, classroom/media or model access."""
import base64
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import time

from scripts.production_result_export import encrypt, identity
from src.api.webvpn import AuthenticationError, WebVPNSession


def inspect():
    run,slot = os.environ['SOURCE_RUN_ID'],int(os.environ['SOURCE_SLOT'])
    identity(run,slot)
    recipient=base64.b64decode(os.environ['RECIPIENT_PUBLIC_KEY'],validate=True)
    if len(recipient)!=32: raise ValueError('Invalid recipient public key')
    result={'authentication_only':True,'source_context_run_id':run,'source_context_slot':slot,
            'attempts':1,'media_requests':0,'model_calls':0,'verified':False}
    vpn=WebVPNSession();stage='webvpn_login';began=time.monotonic()
    try:
        vpn.login();result['webvpn_login_returned']=True
        stage='icourse_authentication'
        vpn.authenticate_icourse(strict=True)
        result.update(verified=True,status='complete')
    except Exception as error:
        result.update(status='failed',failure_stage=stage,failure_type=type(error).__name__)
        if isinstance(error,AuthenticationError): result['failure_reason']=error.reason
        raise
    finally:
        result.update(seconds=time.monotonic()-began,diagnostics=vpn.auth_diagnostics)
        try:vpn.session.close()
        finally:
            out=Path(os.environ['RUNNER_TEMP'])/'qwen-shards'/'out';out.mkdir(parents=True,exist_ok=True,mode=0o700)
            (out/'auth-inspection.enc').write_bytes(encrypt(json.dumps(result).encode(),recipient,run,slot))


if __name__=='__main__':
    try:
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):inspect()
        print('Authentication-only probe verified; encrypted audit saved')
    except Exception as error:
        print(f'Authentication-only probe failed ({type(error).__name__}); encrypted audit saved when available')
        sys.exit(1)
