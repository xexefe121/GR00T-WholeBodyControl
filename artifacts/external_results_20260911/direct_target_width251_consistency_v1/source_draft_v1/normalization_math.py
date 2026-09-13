"""Pre-layer inference algebra only. No checkpoint, Torch, ONNX or ORT import."""
import ast
import numpy as np

def require_schema(features,mean,std):
    if features.ndim!=2 or features.shape[1]!=1323 or features.dtype!=np.float32:
        raise ValueError('Raw public float32[N,1323] required')
    if mean.shape!=(1323,) or std.shape!=(1323,) or mean.dtype!=np.float32 or std.dtype!=np.float32:
        raise ValueError('Original frozen float32 normalization required')
    if not all(np.isfinite(v).all() for v in (features,mean,std)) or np.any(std<=0):
        raise ValueError('Finite inputs and positive frozen std required')

def normalize64(features,mean,std):
    require_schema(features,mean,std)
    # The released source explicitly Casts, Subs, then Divides before MatMul.
    # There is no input clip or float32 rounding after this normalization.
    centered=features.astype(np.float64)-mean.astype(np.float64)
    result=centered/std.astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite source-algebra normalization')
    return result

def verify_export_algebra(source_text):
    tree=ast.parse(source_text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PromotedDirect')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    expected=ast.parse('value=(features.to(torch.float64)-self.feature_mean)/self.feature_std').body[0]
    if len(forward.body)!=4 or ast.dump(forward.body[1],include_attributes=False)!=ast.dump(expected,include_attributes=False):
        raise ValueError('Changed qualified prelayer normalization')
    export=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export_onnx')
    nodes=next(n for n in export.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='nodes' for t in n.targets))
    expected_nodes=ast.parse("nodes=[helper.make_node('Cast',['features'],['features64'],to=TensorProto.DOUBLE),helper.make_node('Sub',['features64','mean'],['center']),helper.make_node('Div',['center','std'],['normalized'])]").body[0]
    if ast.dump(nodes,include_attributes=False)!=ast.dump(expected_nodes,include_attributes=False):
        raise ValueError('Changed exported Cast/Sub/Div graph prefix')
    return dict(public_input='float32',cast='float64',normalization='separate float64 subtraction then division',
                normalization_parameters='original float32 promoted exactly to float64',input_clipping=False,
                cast_after_normalization=False,actual_model_or_runtime_execution=False)
