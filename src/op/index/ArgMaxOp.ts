import IndexOp from "./IndexOp";

export default class ArgMaxOp extends IndexOp {

  update(accum: number, a: number, accumIdx: number, idx: number): [number, number] {
    return a > accum ? [a, idx] : [accum, accumIdx];
  }

}