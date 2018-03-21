import TransformOp from "./TransformOp";

export default class AsinOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.asin(a);
  }

}