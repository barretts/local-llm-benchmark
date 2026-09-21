from .prompts import pack
from .config import digest


def regional_prompt(adapter, tools, system, regions, query, seed, target=61440, tolerance=64):
    """Position each authoritative region using full-template exact token counts."""
    if len(regions)!=3:
        raise ValueError("three authoritative regions required")
    regions=[doc["text"] if isinstance(doc,dict) else doc for doc in regions]
    prefix="Synthetic repository archive. All filler is labelled OBSOLETE EXAMPLE and unrelated. Only the active signed documents named by the request are authoritative.\n"
    positions=[]
    def messages(text):
        return [{"role":"system","content":system},{"role":"user","content":text}]
    tokenize=lambda m:adapter.tokenize(m,tools)
    for index,(fraction,doc) in enumerate(zip((.05,.50,.95),regions)):
        if not isinstance(doc,str) or not doc:
            raise ValueError("empty authoritative document")
        desired=int(target*fraction)
        packed=pack(tokenize,lambda filler:messages(prefix+filler),seed+index*100003,desired,tolerance)
        prefix=packed["messages"][1]["content"]
        position=tokenize(messages(prefix))["count"]
        positions.append({"region":index,"token_position":position,"fraction":position/target,"position_method":"full-template prefix count; assistant prefix included"})
        prefix+="\n"+doc+"\n"
    packed=pack(tokenize,lambda filler:messages(prefix+filler+"\nFINAL REQUEST:\n"+query),seed+900001,target,tolerance)
    packed["region_positions"]=positions
    if any(abs(p["fraction"]-f)>.01 for p,f in zip(positions,(.05,.50,.95))):
        raise ValueError("authoritative region misplaced")
    packed["prompt_hash"]=digest(packed["messages"])
    return packed
