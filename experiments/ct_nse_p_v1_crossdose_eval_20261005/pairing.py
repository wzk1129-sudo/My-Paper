"""Physical pairing and exact DICOM-NPY verification of all 610 pairs."""
from common import *
import pydicom
from natsort import natsorted
def audit_pairs(access):
    rows=[];access.active=True
    series={v:natsorted((RAW/v).glob('*.IMA'),key=str) for v in ('quarter_1mm','full_1mm')}
    assert all(len(x)==610 for x in series.values())
    access.allowed_raw={str(p.resolve()) for xs in series.values() for p in xs}
    z=[]
    for i in range(610):
        ds=[pydicom.dcmread(series[v][i]) for v in ('quarter_1mm','full_1mm')]
        a,b=ds
        assert str(a.PatientID)==str(b.PatientID) and str(a.PatientID)
        assert int(a.InstanceNumber)==int(b.InstanceNumber)==i+1
        geom={}
        for name in ('ImagePositionPatient','ImageOrientationPatient','PixelSpacing'):
            av=np.array(getattr(a,name),float);bv=np.array(getattr(b,name),float)
            assert av.shape==bv.shape and np.allclose(av,bv,atol=1e-6,rtol=0),(i,name)
            geom[name]=av.tolist()
        assert a.Rows==b.Rows==512 and a.Columns==b.Columns==512
        assert float(a.SliceThickness)==float(b.SliceThickness)==1
        assert float(a.RescaleSlope)==float(b.RescaleSlope)==1
        assert float(a.RescaleIntercept)==float(b.RescaleIntercept)==-1024
        assert float(a.SliceLocation)==float(b.SliceLocation)
        files={}
        for dose,d,p in zip(('25','target'),ds,(series['quarter_1mm'][i],series['full_1mm'][i])):
            n=DATA/f'L333_{dose}_{i:03d}_img.npy'
            assert n.is_file() and not n.is_symlink()
            arr=np.load(n,allow_pickle=False)
            assert arr.shape==(512,512) and arr.dtype==np.uint16 and np.isfinite(arr).all()
            assert np.array_equal(d.pixel_array.astype(np.uint16),arr),(i,dose)
            hu=arr.astype(np.float32)-1024;x=normalize(arr)
            files[dose]={'NPY':str(n),'DICOM':str(p),'NPY_sha256':sha(n),'DICOM_sha256':sha(p),
                        'image_shape':[512,512],'dtype':str(arr.dtype),
                        'stored_range':[int(arr.min()),int(arr.max())],
                        'HU_range':[float(hu.min()),float(hu.max())],
                        'normalized_range':[float(x.min()),float(x.max())],'DICOM_NPY_exact':True}
        rows.append({'patient':'L333','slice_index_zero_based':i,'instance':i+1,
                     'physical_geometry':geom,'strict_pairing':True,'files':files,
                     'patient_header_match':True,
                     'study_uid_equal':str(a.StudyInstanceUID)==str(b.StudyInstanceUID),
                     'frame_uid_equal':str(a.FrameOfReferenceUID)==str(b.FrameOfReferenceUID)})
        z.append(geom['ImagePositionPatient'][2])
        if (i+1)%100==0:print('PAIRING',i+1,flush=True)
    assert len(set(z))==610 and np.allclose(np.diff(z),np.diff(z)[0],atol=1e-6)
    result={'status':'PASS','slice_count':610,'index_range':[0,609],
       'physical_z_unique':610,'physical_z_step':float(np.diff(z)[0]),'pairs':rows,
       'preprocessing':'raw.float32-1024 -> clip [-1024,3072] HU -> (HU+1024)/4096',
       'UID_note':'Anonymized full/quarter series can have distinct study/frame UIDs. Pairing is proved by common patient folder/header, instance, physical grid and exact DICOM-NPY pixels.',
       'access_audit':access.report()}
    dump(ROOT/'pairing_audit.json',result)
    return result

if __name__=="__main__":
    access=Access()
    audit_pairs(access)
    dump(ROOT/"PAIRING_ENVIRONMENT.json",{"python":sys.version,"executable":sys.executable,"numpy":np.__version__,"pydicom":pydicom.__version__})
    print("ALL_610_PAIRING_PASS",flush=True)
