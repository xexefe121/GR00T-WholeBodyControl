"""Independent ONNX structure and literal promotion checks; no inference."""
import numpy as np

def exact(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError(name + ': shape/dtype/bytes differ')

def audit_graph(graph, checkpoint, normalization):
    import onnx
    from onnx import numpy_helper, TensorProto
    assert graph.ir_version == 8
    assert [(x.domain, x.version) for x in graph.opset_import] == [('', 17)]
    assert not graph.functions and not graph.training_info
    assert len(graph.graph.input) == len(graph.graph.output) == 1
    for value, name, width in [(graph.graph.input[0], 'features', 1323),
                               (graph.graph.output[0], 'normalized_target', 23)]:
        assert value.name == name
        tensor = value.type.tensor_type
        assert tensor.elem_type == TensorProto.FLOAT and len(tensor.shape.dim) == 2
        assert tensor.shape.dim[0].dim_value == 0 and tensor.shape.dim[1].dim_value == width
    expected = [('Cast', ['features'], ['features64']),
                ('Sub', ['features64', 'mean'], ['center']),
                ('Div', ['center', 'std'], ['normalized'])]
    prior = 'normalized'
    for i in range(3):
        m, a, e = 'm'+str(i), 'a'+str(i), 'e'+str(i)
        expected += [('MatMul', [prior, 'w'+str(i)], [m]),
                     ('Add', [m, 'b'+str(i)], [a])]
        if i < 2:
            negative, exp, elu, positive = ['negative'+str(i), 'exp'+str(i), 'elu_negative'+str(i), 'positive'+str(i)]
            expected += [('Min', [a, 'zero'], [negative]),
                         ('Exp', [negative], [exp]),
                         ('Sub', [exp, 'one'], [elu]),
                         ('Greater', [a, 'zero'], [positive]),
                         ('Where', [positive, a, elu], [e])]
            prior = e
        else:
            prior = a
    expected.append(('Cast', [prior], ['normalized_target']))
    assert len(graph.graph.node) == len(expected) == 20
    for index, (node, (kind, inputs, outputs)) in enumerate(zip(graph.graph.node, expected)):
        assert node.domain == '' and node.op_type == kind
        assert list(node.input) == inputs and list(node.output) == outputs
        if kind == 'Cast':
            assert len(node.attribute) == 1
            attr = node.attribute[0]
            assert attr.name == 'to' and attr.type == onnx.AttributeProto.INT
            assert attr.i == (TensorProto.DOUBLE if index == 0 else TensorProto.FLOAT)
        else:
            assert len(node.attribute) == 0
    tensors = {}
    for item in graph.graph.initializer:
        assert item.name not in tensors
        assert item.data_type == TensorProto.DOUBLE and not item.external_data
        tensors[item.name] = numpy_helper.to_array(item)
        assert np.isfinite(tensors[item.name]).all()
    assert set(tensors) == {'mean','std','zero','one','w0','b0','w1','b1','w2','b2'}
    exact(tensors['zero'], np.array(0., np.float64), 'zero literal')
    exact(tensors['one'], np.array(1., np.float64), 'one literal')
    for key, checkpoint_key, norm_key in [('mean','feature_mean','feature_mean'), ('std','feature_std','feature_std')]:
        original = checkpoint[checkpoint_key].numpy()
        assert original.dtype == np.float32 and original.shape == (1323,)
        exact(original, normalization[norm_key], 'original normalization ' + key)
        exact(tensors[key], original.astype(np.float64), 'promoted normalization ' + key)
    assert np.all(tensors['std'] > 0)
    actor = checkpoint['actor_state']
    assert set(actor) == {str(i)+'.'+suffix for i in (0,2,4) for suffix in ('weight','bias')}
    for layer, source_index, shape in zip(range(3), (0,2,4), [(256,1323),(256,256),(23,256)]):
        weight, bias = [actor[str(source_index)+'.'+suffix].numpy() for suffix in ('weight','bias')]
        assert weight.dtype == bias.dtype == np.float32
        assert weight.shape == shape and bias.shape == (shape[0],)
        exact(tensors['w'+str(layer)], weight.astype(np.float64).T.copy(), 'promoted weight')
        exact(tensors['b'+str(layer)], bias.astype(np.float64), 'promoted bias')
    return dict(graph_nodes=20, literal_arrays=10, same_source_weights=True,
                same_source_normalization=True, explicit_float32_input_and_output=True,
                bounded_double_ELU_equivalent=True, model_calls=0)
