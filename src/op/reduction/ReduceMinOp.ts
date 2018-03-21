import ReductionOp from "./ReductionOp";

export default class ReduceMinOp extends ReductionOp {

  get initialValue() {
    return Number.MAX_VALUE;
  }

  update(accum: number, a: number): number {
    return Math.min(accum, a);
  }

}