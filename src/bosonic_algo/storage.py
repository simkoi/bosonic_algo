"""Self-describing, parameter-addressed results and result discovery."""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Callable
import numpy as np
from .fourier import canonical_json

RESULT_SCHEMA=1


def source_fingerprint() -> str:
    """Hash computational sources to invalidate changed numerical implementations."""
    folder=Path(__file__).parent
    h=hashlib.sha256()
    for name in ("config.py","physics.py","fourier.py","classical.py","tgifs.py"):
        h.update(name.encode())
        h.update((folder/name).read_bytes())
    return h.hexdigest()


def result_identity(inputs: dict) -> tuple[str,str]:
    full=hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()
    return full[:20],full


def _versions() -> dict:
    out={}
    for package in ("numpy","scipy","qiskit","bosonic-qiskit"):
        try:out[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:out[package]=None
    return out


class ResultStore:
    """One directory per run; metadata can be searched without loading arrays."""
    def __init__(self, root: str|Path="results"):
        self.root=Path(root)

    def path(self,method:str,run_id:str)->Path:
        if method not in ("classical","tgifs"):
            raise ValueError("Unknown propagation method")
        return self.root/("quantum" if method=="tgifs" else "classical")/run_id

    def load(self,method:str,run_id:str,expected_fingerprint:str|None=None)->tuple[dict,dict]:
        path=self.path(method,run_id)
        meta=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        if (meta.get("schema_version")!=RESULT_SCHEMA or meta.get("status")!="complete"
            or meta.get("run_id")!=run_id or meta.get("method")!=method):
            raise ValueError("Invalid or incomplete cached result")
        if expected_fingerprint is not None and meta.get("fingerprint")!=expected_fingerprint:
            raise ValueError("Saved calculation does not match requested parameters")
        data={}
        for filename in ("observables.npz","snapshots.npz"):
            with np.load(path/filename,allow_pickle=False) as f:
                data.update({k:f[k] for k in f.files})
        checksum=self._data_checksum(data)
        if checksum!=meta.get("data_checksum"):
            raise ValueError("Cached numerical result failed checksum validation")
        return data,meta

    @staticmethod
    def _data_checksum(data:dict)->str:
        h=hashlib.sha256()
        for key in sorted(data):
            a=np.ascontiguousarray(np.asarray(data[key]))
            h.update(key.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def save(self,method:str,inputs:dict,data:dict, *, replace:bool=False) -> dict:
        run_id,fingerprint=result_identity(inputs)
        target=self.path(method,run_id)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and not replace:
            _,meta=self.load(method,run_id,expected_fingerprint=fingerprint)
            return meta
        temp=Path(tempfile.mkdtemp(prefix=".run-",dir=target.parent))
        try:
            snapshot_keys={"psi_t","snapshot_indices","snapshot_times"}
            arrays={k:np.asarray(v) for k,v in data.items()}
            np.savez_compressed(temp/"observables.npz",**{k:v for k,v in arrays.items() if k not in snapshot_keys})
            np.savez_compressed(temp/"snapshots.npz",**{k:v for k,v in arrays.items() if k in snapshot_keys})
            meta={"schema_version":RESULT_SCHEMA,"status":"complete","method":method,
                  "run_id":run_id,"fingerprint":fingerprint,"inputs":inputs,
                  "data_checksum":self._data_checksum(arrays),
                  "created_utc":datetime.now(timezone.utc).isoformat(),
                  "dependency_versions":_versions(),
                  "git_commit":os.environ.get("GIT_COMMIT")}
            (temp/"metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
            if target.exists():
                # replace=True is explicit; keep the old result until its replacement is complete.
                backup=target.with_name(target.name+".previous")
                if backup.exists():shutil.rmtree(backup)
                os.rename(target,backup)
                try:os.rename(temp,target)
                except Exception:
                    os.rename(backup,target)
                    raise
                shutil.rmtree(backup)
            else:
                os.rename(temp,target)
        finally:
            if temp.exists():shutil.rmtree(temp)
        return meta

    def run_or_load(self,method:str,inputs:dict,calculate:Callable[[],dict],*,force=False)->tuple[dict,dict,bool]:
        run_id,fingerprint=result_identity(inputs)
        folder=self.path(method,run_id)
        if folder.exists() and not force:
            data,meta=self.load(method,run_id,expected_fingerprint=fingerprint)
            return data,meta,True
        data=calculate()
        meta=self.save(method,inputs,data,replace=force)
        return data,meta,False

    def list_results(self,method:str|None=None)->list[dict]:
        methods=[method] if method else ["classical","tgifs"]
        found=[]
        for name in methods:
            folder=self.root/("quantum" if name=="tgifs" else "classical")
            for file in sorted(folder.glob("*/metadata.json")):
                try:
                    meta=json.loads(file.read_text(encoding="utf-8"))
                    if meta.get("status")=="complete" and meta.get("schema_version")==RESULT_SCHEMA:
                        found.append(meta)
                except (OSError,ValueError):continue
        return found

    def find_results(self, *,method=None,fit_id=None,include_cavity=None,potential=None,**parameter_matches)->list[dict]:
        """Search metadata; no large arrays are read.

        Extra parameter_matches use dot paths, e.g. **{'config.tgifs.substeps': 20}.
        """
        found=[]
        for meta in self.list_results(method):
            inputs=meta["inputs"]
            if fit_id is not None and inputs.get("fit_id")!=fit_id:continue
            if potential is not None and inputs.get("potential")!=potential:continue
            if include_cavity is not None and inputs["config"]["cavity"]["enabled"]!=include_cavity:continue
            valid=True
            for dotted,expected in parameter_matches.items():
                item=inputs
                for segment in dotted.split("."):
                    if not isinstance(item,dict) or segment not in item:
                        valid=False;break
                    item=item[segment]
                if not valid or item!=expected:valid=False;break
            if valid:found.append(meta)
        return found
