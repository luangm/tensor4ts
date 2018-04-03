import IndexOp from "./IndexOp";
import Tensor from "../../Tensor";

export default class ArgMinOp extends IndexOp {

  constructor(base: Tensor, result: Tensor, dim: number) {
    super(base, result, dim);
  }

  update(accum: number, a: number, accumIdx: number, idx: number): [number, number] {
    return a > accum ? [accum, accumIdx] : [a, idx];
  }

}