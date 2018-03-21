import ReductionOp from "./ReductionOp";

export default class ReduceMeanOp extends ReductionOp {

  get shouldPostProcess() {
    return true;
  }

  getResult(accum: number, n: number): number {
    return accum / n;
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}