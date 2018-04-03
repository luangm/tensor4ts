import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class ReduceMaxOp extends ReductionOp {

  get initialValue() {
    return -Number.MAX_SAFE_INTEGER;
  }

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  update(accum: number, a: number): number {
    return Math.max(accum, a);
  }

}