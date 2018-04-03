import ReductionOp from "./ReductionOp";
import Tensor from "../../Tensor";

export default class ReduceSumOp extends ReductionOp {

  constructor(base: Tensor, result: Tensor, reducedDims: boolean[]) {
    super(base, result, reducedDims);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}