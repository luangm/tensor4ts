import ReductionOp from "./ReductionOp";

export default class ReduceSumOp extends ReductionOp {

  update(accum: number, a: number): number {
    return accum + a;
  }

}