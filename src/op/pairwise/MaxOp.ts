import PairwiseOp from "./PairwiseOp";

export default class MaxOp extends PairwiseOp {

  body(a: number, b?: number): number {
    return a > b ? a : b;
  }

}