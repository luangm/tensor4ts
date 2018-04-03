import IndexOp from "./IndexOp";
import Tensor from "../../Tensor";

export default class ArgMaxOp extends IndexOp {

  constructor(base: Tensor, result: Tensor, dim: number) {
    super(base, result, dim);
  }

  update(accum: number, a: number, accumIdx: number, idx: number): [number, number] {
    return a > accum ? [a, idx] : [accum, accumIdx];
  }

}