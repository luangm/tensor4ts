import TransformOp from "./TransformOp";

export default class Log1pOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.log1p(a);
  }

}