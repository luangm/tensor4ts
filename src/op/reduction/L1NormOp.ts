import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class L1NormOp extends ReductionOp {

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  body(a: number): number {
    return Math.abs(a);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}