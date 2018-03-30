import ReductionOp from "./ReductionOp";

export default class L2NormOp extends ReductionOp {

  get shouldPostProcess() {
    return true;
  }

  body(a: number, b?: number): number {
    return a * a;
  }

  getResult(accum: number, n: number): number {
    return Math.sqrt(accum);
  }

  update(accum: number, a: number): number {
    return accum + a;
  }

}