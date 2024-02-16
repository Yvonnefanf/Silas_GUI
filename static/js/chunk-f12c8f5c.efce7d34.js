(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-f12c8f5c"],{"2c16":function(t,e,i){"use strict";i.r(e);var n=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"components-container"},[t._m(0),i("div",[i("tinymce",{attrs:{height:300},model:{value:t.content,callback:function(e){t.content=e},expression:"content"}})],1),i("div",{staticClass:"editor-content",domProps:{innerHTML:t._s(t.content)}})])},o=[function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("aside",[t._v(" Rich text is a core feature of the management backend, but at the same time it is a place with lots of pits. In the process of selecting rich texts, I also took a lot of detours. The common rich texts on the market have been basically used, and I finally chose Tinymce. See the more detailed rich text comparison and introduction. "),i("a",{staticClass:"link-type",attrs:{target:"_blank",href:"https://panjiachen.github.io/vue-element-admin-site/feature/component/rich-editor.html"}},[t._v("Documentation")])])}],a=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"tinymce-container",class:{fullscreen:t.fullscreen},style:{width:t.containerWidth}},[i("textarea",{staticClass:"tinymce-textarea",attrs:{id:t.tinymceId}}),t._v(" "),i("div",{staticClass:"editor-custom-btn-container"},[i("editorImage",{staticClass:"editor-upload-btn",attrs:{color:"#1890ff"},on:{successCBK:t.imageSuccessCBK}})],1)])},s=[],c=(i("a9e3"),i("b680"),i("d3b7"),i("ac1f"),i("00b4"),i("159b"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"upload-container"},[i("el-button",{style:{background:t.color,borderColor:t.color},attrs:{icon:"el-icon-upload",size:"mini",type:"primary"},on:{click:function(e){t.dialogVisible=!0}}},[t._v(" upload ")]),i("el-dialog",{attrs:{visible:t.dialogVisible},on:{"update:visible":function(e){t.dialogVisible=e}}},[i("el-upload",{staticClass:"editor-slide-upload",attrs:{multiple:!0,"file-list":t.fileList,"show-file-list":!0,"on-remove":t.handleRemove,"on-success":t.handleSuccess,"before-upload":t.beforeUpload,action:"https://httpbin.org/post","list-type":"picture-card"}},[i("el-button",{attrs:{size:"small",type:"primary"}},[t._v(" Click upload ")])],1),i("el-button",{on:{click:function(e){t.dialogVisible=!1}}},[t._v(" Cancel ")]),i("el-button",{attrs:{type:"primary"},on:{click:t.handleSubmit}},[t._v(" Confirm ")])],1)],1)}),r=[],l=(i("d81d"),i("b64b"),i("3ca3"),i("ddb0"),i("2b3d"),i("9861"),{name:"EditorSlideUpload",props:{color:{type:String,default:"#1890ff"}},data:function(){return{dialogVisible:!1,listObj:{},fileList:[]}},methods:{checkAllSuccess:function(){var t=this;return Object.keys(this.listObj).every((function(e){return t.listObj[e].hasSuccess}))},handleSubmit:function(){var t=this,e=Object.keys(this.listObj).map((function(e){return t.listObj[e]}));this.checkAllSuccess()?(this.$emit("successCBK",e),this.listObj={},this.fileList=[],this.dialogVisible=!1):this.$message("Please wait for all images to be uploaded successfully. If there is a network problem, please refresh the page and upload again!")},handleSuccess:function(t,e){for(var i=e.uid,n=Object.keys(this.listObj),o=0,a=n.length;o<a;o++)if(this.listObj[n[o]].uid===i)return this.listObj[n[o]].url=t.files.file,void(this.listObj[n[o]].hasSuccess=!0)},handleRemove:function(t){for(var e=t.uid,i=Object.keys(this.listObj),n=0,o=i.length;n<o;n++)if(this.listObj[i[n]].uid===e)return void delete this.listObj[i[n]]},beforeUpload:function(t){var e=this,i=window.URL||window.webkitURL,n=t.uid;return this.listObj[n]={},new Promise((function(o,a){var s=new Image;s.src=i.createObjectURL(t),s.onload=function(){e.listObj[n]={hasSuccess:!1,uid:t.uid,width:this.width,height:this.height}},o(!0)}))}}}),u=l,d=(i("3cce"),i("2877")),h=Object(d["a"])(u,c,r,!1,null,"3dae379b",null),m=h.exports,f=["advlist anchor autolink autosave code codesample colorpicker colorpicker contextmenu directionality emoticons fullscreen hr image imagetools insertdatetime link lists media nonbreaking noneditable pagebreak paste preview print save searchreplace spellchecker tabfocus table template textcolor textpattern visualblocks visualchars wordcount"],p=f,b=["searchreplace bold italic underline strikethrough alignleft aligncenter alignright outdent indent  blockquote undo redo removeformat subscript superscript code codesample","hr bullist numlist link image charmap preview anchor pagebreak insertdatetime media table emoticons forecolor backcolor fullscreen"],y=b,g=i("b85c"),v=[];function w(){return window.tinymce}var _=function(t,e){var i=document.getElementById(t),n=e||function(){};if(!i){var o=document.createElement("script");o.src=t,o.id=t,document.body.appendChild(o),v.push(n);var a="onload"in o?s:c;a(o)}function s(e){e.onload=function(){this.onerror=this.onload=null;var t,i=Object(g["a"])(v);try{for(i.s();!(t=i.n()).done;){var n=t.value;n(null,e)}}catch(o){i.e(o)}finally{i.f()}v=null},e.onerror=function(){this.onerror=this.onload=null,n(new Error("Failed to load "+t),e)}}function c(t){t.onreadystatechange=function(){if("complete"===this.readyState||"loaded"===this.readyState){this.onreadystatechange=null;var e,i=Object(g["a"])(v);try{for(i.s();!(e=i.n()).done;){var n=e.value;n(null,t)}}catch(o){i.e(o)}finally{i.f()}v=null}}}i&&n&&(w()?n(null,i):v.push(n))},k=_,C="https://cdn.jsdelivr.net/npm/tinymce-all-in-one@4.9.3/tinymce.min.js",j={name:"Tinymce",components:{editorImage:m},props:{id:{type:String,default:function(){return"vue-tinymce-"+ +new Date+(1e3*Math.random()).toFixed(0)}},value:{type:String,default:""},toolbar:{type:Array,required:!1,default:function(){return[]}},menubar:{type:String,default:"file edit insert view format table"},height:{type:[Number,String],required:!1,default:360},width:{type:[Number,String],required:!1,default:"auto"}},data:function(){return{hasChange:!1,hasInit:!1,tinymceId:this.id,fullscreen:!1,languageTypeList:{en:"en",zh:"zh_CN",es:"es_MX",ja:"ja"}}},computed:{containerWidth:function(){var t=this.width;return/^[\d]+(\.[\d]+)?$/.test(t)?"".concat(t,"px"):t}},watch:{value:function(t){var e=this;!this.hasChange&&this.hasInit&&this.$nextTick((function(){return window.tinymce.get(e.tinymceId).setContent(t||"")}))}},mounted:function(){this.init()},activated:function(){window.tinymce&&this.initTinymce()},deactivated:function(){this.destroyTinymce()},destroyed:function(){this.destroyTinymce()},methods:{init:function(){var t=this;k(C,(function(e){e?t.$message.error(e.message):t.initTinymce()}))},initTinymce:function(){var t=this,e=this;window.tinymce.init({selector:"#".concat(this.tinymceId),language:this.languageTypeList["en"],height:this.height,body_class:"panel-body ",object_resizing:!1,toolbar:this.toolbar.length>0?this.toolbar:y,menubar:this.menubar,plugins:p,end_container_on_empty_block:!0,powerpaste_word_import:"clean",code_dialog_height:450,code_dialog_width:1e3,advlist_bullet_styles:"square",advlist_number_styles:"default",imagetools_cors_hosts:["www.tinymce.com","codepen.io"],default_link_target:"_blank",link_title:!1,nonbreaking_force_tab:!0,init_instance_callback:function(i){e.value&&i.setContent(e.value),e.hasInit=!0,i.on("NodeChange Change KeyUp SetContent",(function(){t.hasChange=!0,t.$emit("input",i.getContent())}))},setup:function(t){t.on("FullscreenStateChanged",(function(t){e.fullscreen=t.state}))},convert_urls:!1})},destroyTinymce:function(){var t=window.tinymce.get(this.tinymceId);this.fullscreen&&t.execCommand("mceFullScreen"),t&&t.destroy()},setContent:function(t){window.tinymce.get(this.tinymceId).setContent(t)},getContent:function(){window.tinymce.get(this.tinymceId).getContent()},imageSuccessCBK:function(t){var e=this;t.forEach((function(t){return window.tinymce.get(e.tinymceId).insertContent('<img class="wscnph" src="'.concat(t.url,'" >'))}))}}},O=j,S=(i("e1d7"),Object(d["a"])(O,a,s,!1,null,"6c71010c",null)),x=S.exports,T={name:"TinymceDemo",components:{Tinymce:x},data:function(){return{content:'<h1 style="text-align: center;">Welcome to the TinyMCE demo!</h1><p style="text-align: center; font-size: 15px;"><img title="TinyMCE Logo" src="//www.tinymce.com/images/glyph-tinymce@2x.png" alt="TinyMCE Logo" width="110" height="97" /><ul>\n        <li>Our <a href="//www.tinymce.com/docs/">documentation</a> is a great resource for learning how to configure TinyMCE.</li><li>Have a specific question? Visit the <a href="https://community.tinymce.com/forum/">Community Forum</a>.</li><li>We also offer enterprise grade support as part of <a href="https://tinymce.com/pricing">TinyMCE premium subscriptions</a>.</li>\n      </ul>'}}},I=T,E=(i("57a8"),Object(d["a"])(I,n,o,!1,null,"1e7d8f0e",null));e["default"]=E.exports},"3cce":function(t,e,i){"use strict";i("5a0d")},"434e":function(t,e,i){},"57a8":function(t,e,i){"use strict";i("e505")},"5a0d":function(t,e,i){},e1d7:function(t,e,i){"use strict";i("434e")},e505:function(t,e,i){}}]);