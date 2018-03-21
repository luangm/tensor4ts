import IndexOp from "./IndexOp";

export default class ArgMinOp extends IndexOp {

  update(accum: number, a: number, accumIdx: number, idx: number): [number, number] {
    return a > accum ? [accum, accumIdx] : [a, idx];
  }

}