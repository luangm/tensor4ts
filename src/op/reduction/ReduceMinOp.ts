import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class ReduceMinOp extends ReductionOp {

  get initialValue() {
    return Number.MAX_VALUE;
  }

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  update(accum: number, a: number): number {
    return Math.min(accum, a);
  }

}