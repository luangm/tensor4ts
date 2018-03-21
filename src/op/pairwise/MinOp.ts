import PairwiseOp from "./PairwiseOp";

export default class MinOp extends PairwiseOp {

  body(a: number, b?: number): number {
    return a > b ? b : a;
  }

}