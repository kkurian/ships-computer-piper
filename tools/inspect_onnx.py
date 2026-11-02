"""Check ONNX voices contain only standard operators: no custom domains, external files or functions."""
import sys, onnx
from onnx.external_data_helper import uses_external_data
STD = {"", "ai.onnx", "ai.onnx.ml"}
for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False); onnx.checker.check_model(m)
    doms = set()
    def walk(g):
        for n in g.node:
            doms.add(n.domain)
            for a in n.attribute:
                if a.g.node: walk(a.g)
                for sg in a.graphs: walk(sg)
    walk(m.graph)
    ext = [t.name for t in m.graph.initializer if uses_external_data(t)]
    print(p.split("/")[-1], "| custom domains:", sorted(doms - STD) or "none", "| external files:", ext or "none", "| functions:", [f.name for f in m.functions] or "none", "| metadata:", len(m.metadata_props))
